// Package ner implements a first-party, zero-dependency multilingual Enterprise
// Domain Named Entity Recognition (NER) and Relation Extraction engine for
// Hive query understanding, signal extraction, and FalkorDB graph matching.
//
// The engine combines:
//  1. A Linear-Chain BIO Viterbi Sequence Decoder with hard BIO transition
//     constraints (B-<cat>, I-<cat>, O) and learned transition/emission weights.
//  2. A longest-span multi-token Gazetteer/Catalog matcher for caller-supplied
//     tenant vocabularies and built-in multilingual domain lexicons across 10+
//     languages (EN, FR, DE, PL, ES, IT, PT, NL, JA, ZH).
//  3. Multilingual prepositional, headword, and syntactic context cues alongside
//     character-shape, hyphen/identifier, numeral/unit, and subword affix features.
//  4. Typed binary relation extraction (compares_with, scoped_to, causes_or_drives,
//     measured_over) between recognized spans.
package ner

import (
	"hash/fnv"
	"math"
	"slices"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"
)

// EntityCategory identifies one of the 8 enterprise domain entity classes.
type EntityCategory string

const (
	CategoryMetric           EntityCategory = "metric"
	CategoryScopeSite        EntityCategory = "scope_site"
	CategoryScopeRegion      EntityCategory = "scope_region"
	CategorySupplierOrg      EntityCategory = "supplier_org"
	CategoryServiceSystem    EntityCategory = "service_system"
	CategoryTimeWindow       EntityCategory = "time_window"
	CategoryComparisonTarget EntityCategory = "comparison_target"
	CategoryNumberValue      EntityCategory = "number_value"
)

// AllCategories lists all 8 domain entity categories in canonical BIO state order.
var AllCategories = []EntityCategory{
	CategoryMetric,
	CategoryScopeSite,
	CategoryScopeRegion,
	CategorySupplierOrg,
	CategoryServiceSystem,
	CategoryTimeWindow,
	CategoryComparisonTarget,
	CategoryNumberValue,
}

// RelationType identifies a typed binary relation between two recognized entity spans.
type RelationType string

const (
	RelationComparesWith   RelationType = "compares_with"
	RelationScopedTo       RelationType = "scoped_to"
	RelationCausesOrDrives RelationType = "causes_or_drives"
	RelationMeasuredOver   RelationType = "measured_over"
)

const (
	labelOutside  = "O"
	numCategories = 8
	numBIOStates  = 1 + numCategories*2
	featureBits   = 14
	featureMask   = (1 << featureBits) - 1
	negInf        = -1e9
)

// bioLabels holds the 17 BIO state names: "O", then "B-<cat>", "I-<cat>" for each category.
var bioLabels = func() [numBIOStates]string {
	var out [numBIOStates]string
	out[0] = labelOutside
	for i, cat := range AllCategories {
		out[1+i*2] = "B-" + string(cat)
		out[2+i*2] = "I-" + string(cat)
	}
	return out
}()

func beginState(catIdx int) int  { return 1 + catIdx*2 }
func insideState(catIdx int) int { return 2 + catIdx*2 }

func categoryIndex(cat EntityCategory) int {
	for i, c := range AllCategories {
		if c == cat {
			return i
		}
	}
	return -1
}

func stateCategory(state int) (EntityCategory, bool, bool) {
	if state <= 0 || state >= numBIOStates {
		return "", false, false
	}
	catIdx := (state - 1) / 2
	isBegin := (state-1)%2 == 0
	return AllCategories[catIdx], isBegin, true
}

// Token is one tokenized segment of the input string with byte offsets [Start, End).
type Token struct {
	Text   string `json:"text"`
	Folded string `json:"folded"`
	Start  int    `json:"start"`
	End    int    `json:"end"`
	Shape  string `json:"shape"`
}

// Span is a recognized domain entity mention with byte offsets [Start, End),
// surface text, normalized canonical entity/concept name, and confidence score.
type Span struct {
	Category   EntityCategory `json:"category"`
	Text       string         `json:"text"`
	Normalized string         `json:"normalized"`
	Start      int            `json:"start"`
	End        int            `json:"end"`
	Confidence float64        `json:"confidence"`
	Source     string         `json:"source,omitempty"`
}

// RelationMention represents a directed semantic relation between two spans in a sentence.
type RelationMention struct {
	Type             RelationType   `json:"type"`
	Source           string         `json:"source"`
	SourceNormalized string         `json:"source_normalized"`
	SourceCategory   EntityCategory `json:"source_category"`
	Target           string         `json:"target"`
	TargetNormalized string         `json:"target_normalized"`
	TargetCategory   EntityCategory `json:"target_category"`
	Confidence       float64        `json:"confidence"`
	Cue              string         `json:"cue,omitempty"`
}

// AnalysisResult contains the full BIO sequence tagger and relation extractor output.
type AnalysisResult struct {
	Tokens    []Token           `json:"tokens"`
	Labels    []string          `json:"labels"`
	Spans     []Span            `json:"spans"`
	Relations []RelationMention `json:"relations"`
	PathScore float64           `json:"path_score"`
}

// SpansByCategory filters recognized spans by category.
func (r AnalysisResult) SpansByCategory(cat EntityCategory) []Span {
	var out []Span
	for _, s := range r.Spans {
		if s.Category == cat {
			out = append(out, s)
		}
	}
	return out
}

// GoldSpan defines a labeled entity span for supervised sequence calibration.
type GoldSpan struct {
	Category EntityCategory `json:"category"`
	Start    int            `json:"start"`
	End      int            `json:"end"`
	Text     string         `json:"text,omitempty"`
}

// TrainingExample is a labeled sentence with optional dynamic catalog entries.
type TrainingExample struct {
	Text    string     `json:"text"`
	Catalog []string   `json:"catalog,omitempty"`
	Spans   []GoldSpan `json:"spans"`
}

// phraseEntry represents a multi-token phrase in the lexicon/gazetteer.
type phraseEntry struct {
	tokens   []string
	category EntityCategory
	weight   float64
}

// Engine is a concurrent-safe Linear-Chain BIO Viterbi NER and Relation Extraction model.
type Engine struct {
	mu          sync.RWMutex
	transitions [numBIOStates][numBIOStates]float64
	weights     [numBIOStates][1 << featureBits]float64
	lexicon     map[string][]phraseEntry // keyed by first folded token
}

var (
	defaultEngineOnce sync.Once
	defaultEngineInst *Engine
)

// DefaultEngine returns the singleton enterprise NER engine pre-calibrated on
// the built-in multilingual seed corpus.
func DefaultEngine() *Engine {
	defaultEngineOnce.Do(func() {
		e := NewEngine()
		_ = e.Train(builtinSeedCorpus())
		defaultEngineInst = e
	})
	return defaultEngineInst
}

// NewEngine constructs a new Enterprise NER engine initialized with valid BIO
// transition constraints and the multilingual domain lexicon.
func NewEngine() *Engine {
	e := &Engine{
		lexicon: make(map[string][]phraseEntry, 256),
	}
	e.initTransitions()
	e.initMultilingualLexicon()
	return e
}

func (e *Engine) initTransitions() {
	for prev := 0; prev < numBIOStates; prev++ {
		for next := 0; next < numBIOStates; next++ {
			if next == 0 {
				// Any state -> O is valid.
				e.transitions[prev][next] = 0.05
				continue
			}
			_, isBegin, _ := stateCategory(next)
			if isBegin {
				// Any state -> B-<cat> is valid, slightly penalizing back-to-back same-category B tags.
				e.transitions[prev][next] = 0.0
				continue
			}
			// next is I-<cat>: valid ONLY when prev is B-<cat> or I-<cat> of the SAME category.
			nextCatIdx := (next - 1) / 2
			if prev == beginState(nextCatIdx) || prev == insideState(nextCatIdx) {
				e.transitions[prev][next] = 0.45
			} else {
				e.transitions[prev][next] = negInf
			}
		}
	}
}

func (e *Engine) addPhrase(phrase string, cat EntityCategory, weight float64) {
	toks := foldTokenSlice(phrase)
	if len(toks) == 0 {
		return
	}
	first := toks[0]
	e.lexicon[first] = append(e.lexicon[first], phraseEntry{
		tokens:   toks,
		category: cat,
		weight:   weight,
	})
}

func (e *Engine) initMultilingualLexicon() {
	// 1. Metrics across EN, FR, DE, PL, ES, IT, PT, NL, JA, ZH
	metrics := []string{
		// FR
		"marge brute", "marge nette", "marge operationnelle", "marge opérationnelle",
		"taux de rupture", "taux de service", "taux de conversion", "taux de retention", "taux de rétention",
		"taux d'erreur", "taux erreur", "taux de retour", "panier moyen", "chiffre d'affaires", "chiffre affaires",
		"delai de livraison", "délai de livraison", "rotation des stocks", "couverture de stock",
		"cout logistique", "coût logistique", "disponibilite produit", "disponibilité produit",
		"retard fournisseur", "delai fournisseur", "délai fournisseur",
		// EN
		"gross margin", "net margin", "operating margin", "stockout rate", "out of stock rate",
		"fill rate", "service level", "churn rate", "retention rate", "conversion rate",
		"error rate", "return rate", "average order value", "order value", "revenue",
		"lead time", "delivery delay", "supplier delay", "inventory turnover", "stock coverage",
		"latency", "p95 latency", "p99 latency", "throughput", "uptime", "availability",
		"otif", "arr", "mrr", "nps", "csat", "ebitda", "roas", "cac", "ltv", "sla",
		// DE
		"bruttomarge", "nettomarge", "ausfallrate", "fehlbestandsquote", "lieferbereitschaft",
		"abwanderungsrate", "fehlerrate", "umsatz", "durchlaufzeit", "lagerumschlag", "warenkorbwert",
		"lieferverzug",
		// PL
		"marza brutto", "marża brutto", "marza netto", "marża netto",
		"wskaznik brakow", "wskaźnik braków", "poziom obslugi", "poziom obsługi",
		"wskaznik odrzucen", "wskaźnik odrzuceń", "sredni koszyk", "średni koszyk",
		"przychody", "czas dostawy", "rotacja zapasow", "rotacja zapasów", "opoznienie dostawcy", "opóźnienie dostawcy",
		// ES
		"margen bruto", "margen neto", "tasa de rotura", "tasa de conversion", "tasa de conversión",
		"tasa de abandono", "tasa de error", "ticket medio", "ingresos", "tiempo de entrega",
		"rotacion de inventario", "rotación de inventario", "retraso proveedor",
		// IT
		"margine lordo", "margine netto", "tasso di rottura", "tasso di abbandono",
		"tasso di errore", "scontrino medio", "fatturato", "tempo di consegna", "ritardo fornitore",
		// PT
		"margem bruta", "margem liquida", "margem líquida", "taxa de ruptura",
		"taxa de conversao", "taxa de conversão", "taxa de erro", "ticket medio", "ticket médio",
		"receita", "prazo de entrega", "atraso fornecedor",
		// NL
		"brutomarge", "nettomarge", "voorraadbreuk", "foutpercentage", "omzet",
		"levertijd", "gemiddelde bestelwaarde",
		// JA
		"粗利率", "純利益率", "欠品率", "解約率", "エラー率", "売上高", "平均客単価", "リードタイム", "在庫回転率", "納入遅延",
		// ZH
		"毛利率", "净利率", "缺货率", "流失率", "错误率", "营业额", "客单价", "交货周期", "库存周转率", "供应商延迟",
	}
	for _, m := range metrics {
		e.addPhrase(m, CategoryMetric, 4.8)
	}

	// 2. Geographic Regions & Markets
	regions := []string{
		"occitanie", "hauts-de-france", "bretagne", "ile-de-france", "île-de-france",
		"auvergne-rhone-alpes", "auvergne-rhône-alpes", "nouvelle-aquitaine", "grand est",
		"pays de la loire", "normandie", "bourgogne-franche-comte", "bourgogne-franche-comté",
		"provence-alpes-cote d'azur", "paca", "corse", "centre-val de loire",
		"emea", "dach", "apac", "latam", "amer", "noram", "benelux", "nordics",
		"north america", "south america", "latin america", "western europe", "eastern europe",
		"central europe", "middle east", "asia pacific",
		"france", "germany", "deutschland", "poland", "polska", "spain", "espana", "españa",
		"italy", "italia", "portugal", "netherlands", "nederland", "belgium", "belgique",
		"switzerland", "suisse", "austria", "uk", "united kingdom", "usa", "united states",
		"canada", "japan", "china", "bayern", "bavaria", "mazowsze", "slask", "śląsk",
		"cataluna", "cataluña", "andalucia", "andalucía", "lombardia", "minho", "algarve",
		"関東", "関西", "中部", "九州", "华东", "华南", "华北", "华中",
	}
	for _, r := range regions {
		e.addPhrase(r, CategoryScopeRegion, 4.6)
	}

	// 3. Time Windows across languages
	timeWindows := []string{
		// FR
		"cette semaine", "semaine derniere", "semaine dernière", "la semaine derniere", "la semaine dernière",
		"ce mois", "mois dernier", "ce trimestre", "trimestre dernier", "cette annee", "cette année",
		"annee derniere", "année dernière", "aujourd'hui", "hier",
		"depuis janvier", "depuis fevrier", "depuis février", "depuis mars", "depuis avril",
		"depuis mai", "depuis juin", "depuis juillet", "depuis aout", "depuis août",
		"depuis septembre", "depuis octobre", "depuis novembre", "depuis decembre", "depuis décembre",
		"30 derniers jours", "7 derniers jours", "14 derniers jours", "90 derniers jours",
		// EN
		"this week", "last week", "this month", "last month", "this quarter", "last quarter",
		"this year", "last year", "today", "yesterday", "year to date", "ytd", "mtd", "qtd",
		"last 7 days", "last 14 days", "last 30 days", "last 60 days", "last 90 days",
		"past 7 days", "past 30 days", "past 90 days",
		"since january", "since february", "since march", "since april", "since may", "since june",
		"since july", "since august", "since september", "since october", "since november", "since december",
		// DE
		"diese woche", "letzte woche", "diesen monat", "letzten monat", "dieses quartal", "letztes quartal",
		"letzte 30 tage", "letzte 7 tage", "seit januar",
		// PL
		"w tym tygodniu", "w zeszlym tygodniu", "w zeszłym tygodniu", "w tym miesiacu", "w tym miesiącu",
		"w tym kwartale", "ostatnie 30 dni", "ostatnie 7 dni", "od stycznia",
		// ES
		"esta semana", "semana pasada", "este mes", "mes pasado", "este trimestre", "trimestre pasado",
		"ultimos 30 dias", "últimos 30 días", "desde enero",
		// IT
		"questa settimana", "settimana scorsa", "questo mese", "mese scorso", "questo trimestre",
		"ultimi 30 giorni", "da gennaio",
		// PT
		"esta semana", "semana passada", "este mes", "este mês", "este trimestre",
		"ultimos 30 dias", "últimos 30 dias", "desde janeiro",
		// NL
		"deze week", "vorige week", "deze maand", "vorige maand", "dit kwartaal",
		"laatste 30 dagen", "sinds januari",
		// JA & ZH
		"今週", "先週", "今月", "先月", "今四半期", "過去30日間", "過去7日間",
		"本周", "上周", "本月", "上月", "本季度", "上季度", "过去30天", "过去7天",
	}
	for _, tw := range timeWindows {
		e.addPhrase(tw, CategoryTimeWindow, 4.7)
	}

	// 4. Known Enterprise Systems & Services
	services := []string{
		"checkout-api", "payment-service", "sap-erp", "wms-core", "auth-gateway",
		"order-service", "inventory-api", "billing-service", "erp-core", "crm-sync",
	}
	for _, svc := range services {
		e.addPhrase(svc, CategoryServiceSystem, 4.9)
	}

	// 5. Known CJK Sites & Suppliers
	cjkSites := []string{"東京店舗", "大阪倉庫", "名古屋工場", "上海仓库", "北京门店", "深圳工厂"}
	for _, s := range cjkSites {
		e.addPhrase(s, CategoryScopeSite, 4.8)
	}
	cjkSuppliers := []string{"東洋物流", "日本サプライ", "华东供应链", "远东物流"}
	for _, s := range cjkSuppliers {
		e.addPhrase(s, CategorySupplierOrg, 4.8)
	}
}

// Analyze tokenizes text, runs the Linear-Chain BIO Viterbi sequence decoder
// with dynamic catalog matching and multilingual contextual cues, and extracts
// typed entity spans and binary relations.
func (e *Engine) Analyze(text string, catalog []string) AnalysisResult {
	tokens := Tokenize(text)
	if len(tokens) == 0 {
		return AnalysisResult{}
	}

	e.mu.RLock()
	emissions := e.computeEmissions(tokens, catalog)
	path, pathScore := e.viterbiDecode(tokens, emissions)
	e.mu.RUnlock()

	labels := make([]string, len(tokens))
	for i, s := range path {
		labels[i] = bioLabels[s]
	}

	spans := extractSpans(text, tokens, path, emissions)
	relations := extractRelations(tokens, spans)

	return AnalysisResult{
		Tokens:    tokens,
		Labels:    labels,
		Spans:     spans,
		Relations: relations,
		PathScore: pathScore,
	}
}

// Train calibrates the engine's linear-chain emission and transition weights on
// domain-labeled examples using structured averaged perceptron updates.
func (e *Engine) Train(examples []TrainingExample) error {
	if len(examples) == 0 {
		return nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()

	const epochs = 6
	const lr = 0.25

	for epoch := 0; epoch < epochs; epoch++ {
		for _, ex := range examples {
			tokens := Tokenize(ex.Text)
			if len(tokens) == 0 {
				continue
			}
			goldPath := goldBIOPath(ex.Text, tokens, ex.Spans)
			emissions := e.computeEmissions(tokens, ex.Catalog)
			predPath, _ := e.viterbiDecode(tokens, emissions)

			same := true
			for i := range goldPath {
				if goldPath[i] != predPath[i] {
					same = false
					break
				}
			}
			if same {
				continue
			}

			tokenFeats := extractTokenFeatures(tokens)
			for i := range tokens {
				gState := goldPath[i]
				pState := predPath[i]
				if gState != pState {
					for _, f := range tokenFeats[i] {
						e.weights[gState][f] += lr
						e.weights[pState][f] -= lr
					}
				}
				if i > 0 {
					gPrev := goldPath[i-1]
					pPrev := predPath[i-1]
					if gPrev != pPrev || gState != pState {
						if e.transitions[gPrev][gState] > negInf/2 {
							e.transitions[gPrev][gState] += lr * 0.2
						}
						if e.transitions[pPrev][pState] > negInf/2 {
							e.transitions[pPrev][pState] -= lr * 0.2
						}
					}
				}
			}
		}
	}
	return nil
}

func goldBIOPath(text string, tokens []Token, spans []GoldSpan) []int {
	path := make([]int, len(tokens))
	for _, sp := range spans {
		catIdx := categoryIndex(sp.Category)
		if catIdx < 0 {
			continue
		}
		start, end := sp.Start, sp.End
		if start == 0 && end == 0 && sp.Text != "" {
			if idx := strings.Index(text, sp.Text); idx >= 0 {
				start = idx
				end = idx + len(sp.Text)
			}
		}
		if end <= start {
			continue
		}
		first := true
		for i, tok := range tokens {
			if tok.Start >= start && tok.End <= end {
				if first {
					path[i] = beginState(catIdx)
					first = false
				} else {
					path[i] = insideState(catIdx)
				}
			}
		}
	}
	return path
}

func (e *Engine) viterbiDecode(tokens []Token, emissions [][numBIOStates]float64) ([]int, float64) {
	n := len(tokens)
	dp := make([][numBIOStates]float64, n)
	back := make([][numBIOStates]int, n)

	// Initialization at t = 0
	for s := 0; s < numBIOStates; s++ {
		if s > 0 && (s-1)%2 == 1 {
			// I-<cat> cannot start a sequence at t = 0.
			dp[0][s] = negInf
		} else {
			dp[0][s] = emissions[0][s]
		}
	}

	// Recursion for t = 1..n-1
	for t := 1; t < n; t++ {
		for next := 0; next < numBIOStates; next++ {
			bestScore := negInf
			bestPrev := 0
			for prev := 0; prev < numBIOStates; prev++ {
				tr := e.transitions[prev][next]
				if tr <= negInf/2 || dp[t-1][prev] <= negInf/2 {
					continue
				}
				score := dp[t-1][prev] + tr + emissions[t][next]
				if score > bestScore {
					bestScore = score
					bestPrev = prev
				}
			}
			dp[t][next] = bestScore
			back[t][next] = bestPrev
		}
	}

	// Termination
	bestFinalState := 0
	bestFinalScore := dp[n-1][0]
	for s := 1; s < numBIOStates; s++ {
		if dp[n-1][s] > bestFinalScore {
			bestFinalScore = dp[n-1][s]
			bestFinalState = s
		}
	}

	path := make([]int, n)
	cur := bestFinalState
	for t := n - 1; t >= 0; t-- {
		path[t] = cur
		cur = back[t][cur]
	}
	return path, bestFinalScore
}

func (e *Engine) computeEmissions(tokens []Token, catalog []string) [][numBIOStates]float64 {
	n := len(tokens)
	emissions := make([][numBIOStates]float64, n)
	tokenFeats := extractTokenFeatures(tokens)

	// Base outside bias + learned feature weights
	for i := 0; i < n; i++ {
		emissions[i][0] = 1.0
		for s := 0; s < numBIOStates; s++ {
			var w float64
			for _, f := range tokenFeats[i] {
				w += e.weights[s][f]
			}
			emissions[i][s] += w
		}
	}

	// 1. Dynamic Caller Catalog longest-span matching
	if len(catalog) > 0 {
		catPhrases := buildCatalogPhrases(catalog)
		applyPhraseMatches(tokens, catPhrases, &emissions)
	}

	// 2. Built-in Multilingual Gazetteer longest-span matching
	e.applyLexiconMatches(tokens, &emissions)

	// 3. Contextual syntactic & character-shape rules across 10+ languages
	applyContextualCues(tokens, &emissions)

	return emissions
}

func buildCatalogPhrases(catalog []string) map[string][]phraseEntry {
	out := make(map[string][]phraseEntry, len(catalog)*2)
	for _, raw := range catalog {
		raw = strings.TrimSpace(raw)
		if raw == "" {
			continue
		}
		cat := inferCatalogCategory(raw)
		for _, variant := range []string{raw, strings.ReplaceAll(raw, "_", " ")} {
			toks := foldTokenSlice(variant)
			if len(toks) == 0 {
				continue
			}
			first := toks[0]
			out[first] = append(out[first], phraseEntry{
				tokens:   toks,
				category: cat,
				weight:   5.2,
			})
		}
	}
	return out
}

func inferCatalogCategory(entry string) EntityCategory {
	folded := foldString(entry)
	tokens := strings.Fields(folded)
	if len(tokens) > 0 {
		first := tokens[0]
		if siteHeadwords[first] || hasSitePrefix(folded) {
			return CategoryScopeSite
		}
		if supplierHeadwords[first] {
			return CategorySupplierOrg
		}
		if looksLikeServiceIdentifier(folded) {
			return CategoryServiceSystem
		}
	}
	return CategoryMetric
}

func applyPhraseMatches(tokens []Token, table map[string][]phraseEntry, emissions *[][numBIOStates]float64) {
	n := len(tokens)
	for i := 0; i < n; i++ {
		entries := table[tokens[i].Folded]
		for _, pe := range entries {
			m := len(pe.tokens)
			if i+m > n {
				continue
			}
			matched := true
			for k := 1; k < m; k++ {
				if tokens[i+k].Folded != pe.tokens[k] {
					matched = false
					break
				}
			}
			if !matched {
				continue
			}
			catIdx := categoryIndex(pe.category)
			if catIdx < 0 {
				continue
			}
			bonus := pe.weight + float64(m)*0.35
			(*emissions)[i][beginState(catIdx)] += bonus
			for k := 1; k < m; k++ {
				(*emissions)[i+k][insideState(catIdx)] += bonus
			}
		}
	}
}

func (e *Engine) applyLexiconMatches(tokens []Token, emissions *[][numBIOStates]float64) {
	applyPhraseMatches(tokens, e.lexicon, emissions)
}

var (
	siteHeadwords = map[string]bool{
		"magasin": true, "magasins": true, "entrepot": true, "entrepots": true,
		"site": true, "usine": true, "depot": true, "agence": true, "boutique": true,
		"store": true, "stores": true, "warehouse": true, "warehouses": true,
		"plant": true, "facility": true, "branch": true, "outlet": true, "hub": true, "node": true,
		"filiale": true, "lager": true, "werk": true, "standort": true, "niederlassung": true,
		"sklep": true, "sklepie": true, "magazyn": true, "magazynie": true, "zaklad": true, "oddzial": true,
		"tienda": true, "almacen": true, "planta": true, "sucursal": true,
		"negozio": true, "magazzino": true, "stabilimento": true,
		"loja": true, "armazem": true, "fabrica": true,
		"winkel": true, "magazijn": true, "vestiging": true, "filiaal": true,
		"店舗": true, "倉庫": true, "工場": true, "拠点": true,
		"门店": true, "仓库": true, "工厂": true, "站点": true,
	}

	supplierHeadwords = map[string]bool{
		"fournisseur": true, "fournisseurs": true, "partenaire": true, "partenaires": true,
		"prestataire": true, "transporteur": true, "sous-traitant": true,
		"supplier": true, "suppliers": true, "vendor": true, "vendors": true,
		"partner": true, "partners": true, "carrier": true, "distributor": true, "contractor": true,
		"lieferant": true, "lieferanten": true, "spediteur": true, "hersteller": true,
		"dostawca": true, "dostawcy": true, "przewoznik": true,
		"proveedor": true, "proveedores": true, "socio": true, "transportista": true,
		"fornitore": true, "fornitori": true, "corriere": true,
		"fornecedor": true, "fornecedores": true, "parceiro": true, "transportadora": true,
		"leverancier": true, "leveranciers": true, "vervoerder": true,
		"サプライヤー": true, "仕入先": true, "ベンダー": true, "取引先": true,
		"供应商": true, "厂商": true, "合作伙伴": true, "承运商": true,
	}

	connectingParticles = map[string]bool{
		"de": true, "du": true, "des": true, "d": true, "of": true, "at": true, "in": true, "nr": true, "no": true,
	}

	regionPrepositions = map[string]bool{
		"en": true, "dans": true, "in": true, "na": true, "w": true, "em": true, "nel": true, "nella": true,
	}

	monthNames = map[string]bool{
		"janvier": true, "fevrier": true, "mars": true, "avril": true, "mai": true, "juin": true,
		"juillet": true, "aout": true, "septembre": true, "octobre": true, "novembre": true, "decembre": true,
		"january": true, "february": true, "march": true, "april": true, "may": true, "june": true,
		"july": true, "august": true, "september": true, "october": true, "november": true, "december": true,
		"januar": true, "februar": true, "marz": true, "juni": true, "juli": true, "oktober": true, "dezember": true,
		"enero": true, "febrero": true, "marzo": true, "abril": true, "mayo": true, "junio": true, "julio": true, "agosto": true, "septiembre": true, "octubre": true, "noviembre": true, "diciembre": true,
	}

	unitTokens = map[string]bool{
		"%": true, "€": true, "$": true, "£": true, "¥": true, "zł": true, "zl": true,
		"eur": true, "usd": true, "gbp": true, "chf": true, "jpy": true, "cny": true, "pln": true,
		"ms": true, "s": true, "sec": true, "min": true, "h": true, "hr": true, "hrs": true,
		"pts": true, "pt": true, "bps": true, "bp": true, "k€": true, "m€": true, "k$": true, "m$": true,
	}

	stopOrGrammarTokens = map[string]bool{
		"pourquoi": true, "comment": true, "quel": true, "quelle": true, "quels": true, "quelles": true,
		"est": true, "sont": true, "le": true, "la": true, "les": true, "un": true, "une": true,
		"au": true, "aux": true, "chez": true, "sur": true, "pour": true, "avec": true, "sans": true,
		"et": true, "ou": true, "mais": true, "impacte": true, "affecte": true, "cause": true,
		"why": true, "how": true, "what": true, "which": true, "when": true, "where": true,
		"is": true, "are": true, "was": true, "were": true, "the": true, "a": true, "an": true,
		"and": true, "or": true, "for": true, "with": true, "from": true, "by": true, "to": true,
		"warum": true, "wie": true, "der": true, "die": true, "das": true, "und": true,
		"dlaczego": true, "jak": true, "czy": true, "oraz": true,
		"por": true, "que": true, "como": true, "el": true, "los": true, "las": true,
		"perche": true, "il": true, "gli": true, "waarom": true, "hoe": true, "het": true,
	}
)

func hasSitePrefix(folded string) bool {
	for _, p := range []string{"warehouse-", "store-", "site-", "plant-", "depot-", "entrepot-", "magasin-", "dc-", "node-"} {
		if strings.HasPrefix(folded, p) {
			return true
		}
	}
	return false
}

func looksLikeServiceIdentifier(folded string) bool {
	if hasSitePrefix(folded) {
		return false
	}
	if !strings.Contains(folded, "-") {
		return false
	}
	for _, affix := range []string{
		"-api", "-service", "-svc", "-erp", "-wms", "-crm", "-core", "-gateway",
		"-gw", "-db", "-cache", "-worker", "-pipeline", "-proxy", "-ingress",
		"-cluster", "-engine", "-broker", "-queue", "-auth",
		"api-", "svc-", "wms-", "erp-", "sap-", "crm-", "auth-", "payment-", "checkout-",
	} {
		if strings.Contains(folded, affix) {
			return true
		}
	}
	return false
}

func isProperOrIdentifierToken(tok Token) bool {
	if stopOrGrammarTokens[tok.Folded] {
		return false
	}
	if strings.HasPrefix(tok.Text, "#") && len(tok.Text) > 1 {
		return true
	}
	if tok.Shape == "title" || tok.Shape == "upper" || tok.Shape == "mixed" || tok.Shape == "id" {
		return true
	}
	if tok.Shape == "digit" {
		return true
	}
	return false
}

func applyContextualCues(tokens []Token, emissions *[][numBIOStates]float64) {
	n := len(tokens)
	siteIdx := categoryIndex(CategoryScopeSite)
	regIdx := categoryIndex(CategoryScopeRegion)
	supIdx := categoryIndex(CategorySupplierOrg)
	svcIdx := categoryIndex(CategoryServiceSystem)
	timeIdx := categoryIndex(CategoryTimeWindow)
	cmpIdx := categoryIndex(CategoryComparisonTarget)
	numIdx := categoryIndex(CategoryNumberValue)

	consumedByComparison := make([]bool, n)

	// 1. Comparison Target cues ("par rapport à <Target>", "vs <Target>", "contre <Target>", "compared to <Target>")
	for i := 0; i < n; i++ {
		cueLen, targetLen := matchComparisonCue(tokens, i)
		if cueLen > 0 && targetLen > 0 {
			total := cueLen + targetLen
			(*emissions)[i][beginState(cmpIdx)] += 7.5
			for k := 1; k < total; k++ {
				(*emissions)[i+k][insideState(cmpIdx)] += 7.5
			}
			for k := 0; k < total; k++ {
				consumedByComparison[i+k] = true
			}
			i += total - 1
		}
	}

	// 2. Site headword + optional particle + ProperName/#ID ("magasin de Lille", "store #42", "entrepôt Lyon", "site de Toulouse")
	//    and standalone site identifiers ("warehouse-eu-1")
	for i := 0; i < n; i++ {
		if consumedByComparison[i] {
			continue
		}
		f := tokens[i].Folded
		if hasSitePrefix(f) {
			(*emissions)[i][beginState(siteIdx)] += 6.5
			continue
		}
		if siteHeadwords[f] {
			if i+1 < n && !consumedByComparison[i+1] {
				if connectingParticles[tokens[i+1].Folded] && i+2 < n && isProperOrIdentifierToken(tokens[i+2]) {
					(*emissions)[i][beginState(siteIdx)] += 6.5
					(*emissions)[i+1][insideState(siteIdx)] += 6.5
					(*emissions)[i+2][insideState(siteIdx)] += 6.5
					continue
				}
				if isProperOrIdentifierToken(tokens[i+1]) {
					(*emissions)[i][beginState(siteIdx)] += 6.5
					(*emissions)[i+1][insideState(siteIdx)] += 6.5
					continue
				}
			}
		}
	}

	// 3. Supplier headword + ProperName/#ID ("fournisseur BoisNord", "supplier AcmeCorp", "vendor #108", "partenaire LogisFrance")
	for i := 0; i < n; i++ {
		if consumedByComparison[i] {
			continue
		}
		if supplierHeadwords[tokens[i].Folded] {
			// Avoid triggering when preceded by "retard"/"délai" with no proper identifier after.
			if i+1 < n && !consumedByComparison[i+1] {
				if connectingParticles[tokens[i+1].Folded] && i+2 < n && isProperOrIdentifierToken(tokens[i+2]) {
					(*emissions)[i][beginState(supIdx)] += 6.6
					(*emissions)[i+1][insideState(supIdx)] += 6.6
					(*emissions)[i+2][insideState(supIdx)] += 6.6
					continue
				}
				if isProperOrIdentifierToken(tokens[i+1]) && !stopOrGrammarTokens[tokens[i+1].Folded] {
					(*emissions)[i][beginState(supIdx)] += 6.6
					(*emissions)[i+1][insideState(supIdx)] += 6.6
					continue
				}
			}
		}
	}

	// 4. Service/System identifiers ("checkout-api", "payment-service", "SAP-ERP", "wms-core", "auth-gateway")
	for i := 0; i < n; i++ {
		if consumedByComparison[i] {
			continue
		}
		if looksLikeServiceIdentifier(tokens[i].Folded) {
			(*emissions)[i][beginState(svcIdx)] += 6.5
		}
	}

	// 5. Time window patterns ("Q3 2026", "T2 2025", "last 30 days", "depuis janvier")
	consumedByTime := make([]bool, n)
	for i := 0; i < n; i++ {
		if consumedByComparison[i] {
			continue
		}
		f := tokens[i].Folded
		if isQuarterToken(f) {
			(*emissions)[i][beginState(timeIdx)] += 5.8
			consumedByTime[i] = true
			if i+1 < n && isYearToken(tokens[i+1].Folded) {
				(*emissions)[i+1][insideState(timeIdx)] += 5.8
				consumedByTime[i+1] = true
			}
			continue
		}
		if (f == "depuis" || f == "since" || f == "seit" || f == "od" || f == "desde" || f == "da" || f == "sinds") &&
			i+1 < n && (monthNames[tokens[i+1].Folded] || isYearToken(tokens[i+1].Folded)) {
			(*emissions)[i][beginState(timeIdx)] += 6.0
			(*emissions)[i+1][insideState(timeIdx)] += 6.0
			consumedByTime[i] = true
			consumedByTime[i+1] = true
			continue
		}
		if (f == "last" || f == "past" || f == "letzte" || f == "ostatnie" || f == "ultimos" || f == "ultimi" || f == "laatste") &&
			i+2 < n && tokens[i+1].Shape == "digit" && isDurationUnit(tokens[i+2].Folded) {
			(*emissions)[i][beginState(timeIdx)] += 6.2
			(*emissions)[i+1][insideState(timeIdx)] += 6.2
			(*emissions)[i+2][insideState(timeIdx)] += 6.2
			consumedByTime[i], consumedByTime[i+1], consumedByTime[i+2] = true, true, true
		}
	}

	// 6. Prepositional Region cue ("en Occitanie", "en Hauts-de-France", "in EMEA")
	for i := 0; i+1 < n; i++ {
		if consumedByComparison[i] || consumedByComparison[i+1] || consumedByTime[i+1] {
			continue
		}
		if regionPrepositions[tokens[i].Folded] {
			next := tokens[i+1]
			if (next.Shape == "title" || next.Shape == "upper") &&
				!monthNames[next.Folded] && !siteHeadwords[next.Folded] &&
				!supplierHeadwords[next.Folded] && !stopOrGrammarTokens[next.Folded] &&
				!isQuarterToken(next.Folded) {
				(*emissions)[i+1][beginState(regIdx)] += 3.8
			}
		}
	}

	// 7. Number value + optional thousand groups + unit/currency/percentage ("38.5%", "10 000 €", "250ms", "-4.2 pts")
	for i := 0; i < n; i++ {
		if consumedByComparison[i] || consumedByTime[i] {
			continue
		}
		// Skip numbers preceded by site or supplier headwords ("store #42", "vendor #108")
		if i > 0 && (siteHeadwords[tokens[i-1].Folded] || supplierHeadwords[tokens[i-1].Folded]) {
			continue
		}
		if isNumberWithUnitToken(tokens[i].Text) {
			(*emissions)[i][beginState(numIdx)] += 6.2
			continue
		}
		if tokens[i].Shape == "digit" {
			end := i
			for end+1 < n && tokens[end+1].Shape == "digit" && len(tokens[end+1].Text) == 3 {
				end++
			}
			if end+1 < n && unitTokens[tokens[end+1].Folded] {
				end++
				(*emissions)[i][beginState(numIdx)] += 6.2
				for k := i + 1; k <= end; k++ {
					(*emissions)[k][insideState(numIdx)] += 6.2
				}
				i = end
			}
		}
	}
}

func matchComparisonCue(tokens []Token, i int) (cueLen int, targetLen int) {
	n := len(tokens)
	f0 := tokens[i].Folded

	// 3-token cues: "par rapport à/au/aux", "im vergleich zu", "w porownaniu z/do", "em relacao a", "ten opzichte van"
	if i+3 < n {
		f1, f2 := tokens[i+1].Folded, tokens[i+2].Folded
		if (f0 == "par" && f1 == "rapport" && (f2 == "a" || f2 == "au" || f2 == "aux")) ||
			(f0 == "im" && f1 == "vergleich" && f2 == "zu") ||
			(f0 == "w" && f1 == "porownaniu" && (f2 == "z" || f2 == "do")) ||
			(f0 == "em" && f1 == "relacao" && f2 == "a") ||
			(f0 == "ten" && f1 == "opzichte" && f2 == "van") {
			cueLen = 3
		}
	}
	// 2-token cues: "compared to/with", "compare a/au", "relative to", "respecto a", "comparado con/com", "rispetto a", "vergeleken met"
	if cueLen == 0 && i+2 < n {
		f1 := tokens[i+1].Folded
		if (f0 == "compared" && (f1 == "to" || f1 == "with")) ||
			(f0 == "compare" && (f1 == "a" || f1 == "au")) ||
			(f0 == "relative" && f1 == "to") ||
			(f0 == "respecto" && f1 == "a") ||
			(f0 == "comparado" && (f1 == "con" || f1 == "com")) ||
			(f0 == "rispetto" && f1 == "a") ||
			(f0 == "vergeleken" && f1 == "met") {
			cueLen = 2
		}
	}
	// 1-token cues: "vs", "versus", "contre", "gegenuber"
	if cueLen == 0 && i+1 < n {
		if f0 == "vs" || f0 == "versus" || f0 == "contre" || f0 == "gegenuber" {
			cueLen = 1
		}
	}
	if cueLen == 0 {
		return 0, 0
	}

	targetStart := i + cueLen
	if targetStart >= n || tokens[targetStart].Shape == "punct" {
		return 0, 0
	}
	targetLen = 1
	// Allow 2-token comparison targets like "Q2 2026", "last quarter", "semaine dernière", "North America"
	if targetStart+1 < n {
		t0 := tokens[targetStart]
		t1 := tokens[targetStart+1]
		if (isQuarterToken(t0.Folded) && isYearToken(t1.Folded)) ||
			((t0.Folded == "last" || t0.Folded == "this" || t0.Folded == "previous") && isDurationUnit(t1.Folded)) ||
			(t0.Folded == "semaine" && t1.Folded == "derniere") ||
			(t0.Shape == "title" && t1.Shape == "title") {
			targetLen = 2
		}
	}
	return cueLen, targetLen
}

func isQuarterToken(f string) bool {
	switch f {
	case "q1", "q2", "q3", "q4", "t1", "t2", "t3", "t4", "fy25", "fy26", "fy2025", "fy2026":
		return true
	}
	return false
}

func isYearToken(f string) bool {
	if len(f) != 4 {
		return false
	}
	return strings.HasPrefix(f, "202") || strings.HasPrefix(f, "203")
}

func isDurationUnit(f string) bool {
	switch f {
	case "days", "day", "weeks", "week", "months", "month", "quarters", "quarter", "year", "years",
		"jours", "semaines", "mois", "trimestre", "tage", "dni", "dias", "giorni", "dagen":
		return true
	}
	return false
}

func isNumberWithUnitToken(text string) bool {
	if len(text) < 2 {
		return false
	}
	hasDigit := false
	hasUnit := false
	for i, r := range text {
		if unicode.IsDigit(r) {
			hasDigit = true
			continue
		}
		if i == 0 && (r == '-' || r == '+') {
			continue
		}
		if r == '.' || r == ',' {
			continue
		}
		if r == '%' || r == '€' || r == '$' || r == '£' || r == '¥' || unicode.IsLetter(r) {
			if hasDigit {
				hasUnit = true
			} else {
				return false
			}
		}
	}
	return hasDigit && hasUnit
}

func extractSpans(text string, tokens []Token, path []int, emissions [][numBIOStates]float64) []Span {
	var out []Span
	var cur *Span
	for i, state := range path {
		cat, isBegin, ok := stateCategory(state)
		if !ok {
			if cur != nil {
				finalizeSpan(text, cur)
				out = append(out, *cur)
				cur = nil
			}
			continue
		}
		conf := sigmoidConfidence(emissions[i][state])
		if !isBegin && cur != nil && cur.Category == cat {
			cur.End = tokens[i].End
			if conf > cur.Confidence {
				cur.Confidence = conf
			}
			continue
		}
		if cur != nil {
			finalizeSpan(text, cur)
			out = append(out, *cur)
		}
		cur = &Span{
			Category:   cat,
			Start:      tokens[i].Start,
			End:        tokens[i].End,
			Confidence: conf,
			Source:     "viterbi_bio",
		}
	}
	if cur != nil {
		finalizeSpan(text, cur)
		out = append(out, *cur)
	}
	return out
}

func sigmoidConfidence(score float64) float64 {
	c := 1.0 / (1.0 + math.Exp(-0.45*score))
	if c < 0.55 {
		c = 0.55
	}
	if c > 0.99 {
		c = 0.99
	}
	return math.Round(c*100) / 100
}

func finalizeSpan(fullText string, sp *Span) {
	sp.Text = strings.TrimSpace(fullText[sp.Start:sp.End])
	sp.Normalized = normalizeSpanText(sp.Category, sp.Text)
}

func normalizeSpanText(cat EntityCategory, raw string) string {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return ""
	}
	switch cat {
	case CategoryScopeSite:
		// Strip leading site headwords ("magasin de Lille" -> "Lille", "entrepôt Lyon" -> "Lyon", "site de Toulouse" -> "Toulouse"),
		// while preserving "#42" forms ("store #42" -> "store #42") and hyphenated IDs ("warehouse-eu-1" -> "warehouse-eu-1").
		parts := strings.Fields(trimmed)
		if len(parts) >= 2 && siteHeadwords[foldString(parts[0])] {
			restIdx := 1
			if restIdx < len(parts) && connectingParticles[foldString(parts[restIdx])] {
				restIdx++
			}
			if restIdx < len(parts) {
				rem := strings.Join(parts[restIdx:], " ")
				if !strings.HasPrefix(rem, "#") && !isAllDigits(rem) {
					return rem
				}
			}
		}
		return trimmed
	case CategorySupplierOrg:
		// Strip leading supplier headwords ("fournisseur BoisNord" -> "BoisNord", "supplier AcmeCorp" -> "AcmeCorp"),
		// while preserving "#108" forms ("vendor #108" -> "vendor #108").
		parts := strings.Fields(trimmed)
		if len(parts) >= 2 && supplierHeadwords[foldString(parts[0])] {
			restIdx := 1
			if restIdx < len(parts) && connectingParticles[foldString(parts[restIdx])] {
				restIdx++
			}
			if restIdx < len(parts) {
				rem := strings.Join(parts[restIdx:], " ")
				if !strings.HasPrefix(rem, "#") && !isAllDigits(rem) {
					return rem
				}
			}
		}
		return trimmed
	case CategoryComparisonTarget:
		// Strip leading comparison cue ("par rapport à Bretagne" -> "Bretagne", "vs Q2" -> "Q2").
		lower := foldString(trimmed)
		for _, prefix := range []string{
			"par rapport a ", "par rapport au ", "par rapport aux ",
			"im vergleich zu ", "w porownaniu z ", "w porownaniu do ",
			"em relacao a ", "ten opzichte van ",
			"compared to ", "compared with ", "compare a ", "compare au ",
			"relative to ", "respecto a ", "comparado con ", "comparado com ",
			"rispetto a ", "vergeleken met ",
			"versus ", "contre ", "gegenuber ", "vs. ", "vs ",
		} {
			if strings.HasPrefix(lower, prefix) {
				// Return the slice of original words corresponding to the remainder.
				prefixWords := len(strings.Fields(prefix))
				origWords := strings.Fields(trimmed)
				if len(origWords) > prefixWords {
					return strings.Join(origWords[prefixWords:], " ")
				}
			}
		}
		return trimmed
	default:
		return trimmed
	}
}

func isAllDigits(s string) bool {
	for _, r := range s {
		if !unicode.IsDigit(r) {
			return false
		}
	}
	return len(s) > 0
}

// extractRelations derives typed binary relations (compares_with, scoped_to,
// causes_or_drives, measured_over) between recognized spans in the sentence.
func extractRelations(tokens []Token, spans []Span) []RelationMention {
	if len(spans) < 2 {
		return nil
	}
	var out []RelationMention
	seen := make(map[string]bool)
	addRel := func(rel RelationMention) {
		if rel.SourceNormalized == "" || rel.TargetNormalized == "" || rel.SourceNormalized == rel.TargetNormalized {
			return
		}
		key := string(rel.Type) + "|" + rel.SourceNormalized + "|" + rel.TargetNormalized
		if seen[key] {
			return
		}
		seen[key] = true
		out = append(out, rel)
	}

	var metrics, scopes, times, comparisons []Span
	for _, sp := range spans {
		switch sp.Category {
		case CategoryMetric:
			metrics = append(metrics, sp)
		case CategoryScopeSite, CategoryScopeRegion, CategorySupplierOrg, CategoryServiceSystem:
			scopes = append(scopes, sp)
		case CategoryTimeWindow:
			times = append(times, sp)
		case CategoryComparisonTarget:
			comparisons = append(comparisons, sp)
		}
	}

	// 1. RelationComparesWith: link comparison_target to its primary scope, time_window, or metric.
	for _, cmp := range comparisons {
		var anchor *Span
		for i := range scopes {
			if scopes[i].End <= cmp.Start {
				anchor = &scopes[i]
			}
		}
		if anchor == nil && len(times) > 0 {
			anchor = &times[0]
		}
		if anchor == nil && len(metrics) > 0 {
			anchor = &metrics[0]
		}
		if anchor != nil {
			addRel(RelationMention{
				Type:             RelationComparesWith,
				Source:           anchor.Text,
				SourceNormalized: anchor.Normalized,
				SourceCategory:   anchor.Category,
				Target:           cmp.Text,
				TargetNormalized: cmp.Normalized,
				TargetCategory:   cmp.Category,
				Confidence:       0.91,
				Cue:              "comparison",
			})
		}
	}

	// 2. RelationScopedTo: associate each scope with its nearest preceding metric (or nearest metric).
	if len(metrics) > 0 && len(scopes) > 0 {
		for _, sc := range scopes {
			bestMetric := metrics[0]
			bestDist := absInt(sc.Start - bestMetric.End)
			for _, m := range metrics {
				if m.End <= sc.Start {
					dist := sc.Start - m.End
					if bestMetric.End > sc.Start || dist < bestDist {
						bestMetric = m
						bestDist = dist
					}
				} else if bestMetric.End > sc.Start {
					dist := m.Start - sc.End
					if dist < bestDist {
						bestMetric = m
						bestDist = dist
					}
				}
			}
			addRel(RelationMention{
				Type:             RelationScopedTo,
				Source:           bestMetric.Text,
				SourceNormalized: bestMetric.Normalized,
				SourceCategory:   bestMetric.Category,
				Target:           sc.Text,
				TargetNormalized: sc.Normalized,
				TargetCategory:   sc.Category,
				Confidence:       0.90,
				Cue:              "scope_binding",
			})
		}
	}

	// 3. RelationMeasuredOver: associate metrics with recognized time windows.
	for _, m := range metrics {
		for _, tw := range times {
			addRel(RelationMention{
				Type:             RelationMeasuredOver,
				Source:           m.Text,
				SourceNormalized: m.Normalized,
				SourceCategory:   m.Category,
				Target:           tw.Text,
				TargetNormalized: tw.Normalized,
				TargetCategory:   tw.Category,
				Confidence:       0.92,
				Cue:              "time_binding",
			})
		}
	}

	// 4. RelationCausesOrDrives: detect forward ("impacte", "drives", "entraîne", "causes")
	//    or reverse ("causé par", "dû à", "due to", "driven by") causal cues.
	hasForwardCue, hasReverseCue, cueWord := detectCausalCues(tokens)
	if (hasForwardCue || hasReverseCue) && len(metrics) >= 2 {
		src, tgt := metrics[0], metrics[1]
		if hasReverseCue && !hasForwardCue {
			src, tgt = metrics[1], metrics[0]
		}
		addRel(RelationMention{
			Type:             RelationCausesOrDrives,
			Source:           src.Text,
			SourceNormalized: src.Normalized,
			SourceCategory:   src.Category,
			Target:           tgt.Text,
			TargetNormalized: tgt.Normalized,
			TargetCategory:   tgt.Category,
			Confidence:       0.88,
			Cue:              cueWord,
		})
	} else if (hasForwardCue || hasReverseCue) && len(metrics) == 1 && len(scopes) >= 1 {
		m := metrics[0]
		sc := scopes[0]
		if hasReverseCue && sc.Start > m.Start {
			addRel(RelationMention{
				Type:             RelationCausesOrDrives,
				Source:           sc.Text,
				SourceNormalized: sc.Normalized,
				SourceCategory:   sc.Category,
				Target:           m.Text,
				TargetNormalized: m.Normalized,
				TargetCategory:   m.Category,
				Confidence:       0.85,
				Cue:              cueWord,
			})
		}
	}

	return out
}

func detectCausalCues(tokens []Token) (forward bool, reverse bool, cue string) {
	for i, tok := range tokens {
		f := tok.Folded
		switch f {
		case "impacte", "impact", "impacts", "affecte", "affects", "entraine", "drives",
			"provoque", "beeinflusst", "verursacht", "wplywa", "powoduje", "impacta", "impatta":
			return true, false, tok.Text
		case "pourquoi", "why", "warum", "dlaczego", "perche", "waarom", "なぜ", "为什么", "為什麼":
			forward = true
			if cue == "" {
				cue = tok.Text
			}
		case "du", "due", "cause", "caused", "driven", "entraine_par":
			if i+1 < len(tokens) {
				next := tokens[i+1].Folded
				if next == "a" || next == "au" || next == "to" || next == "by" || next == "par" {
					return false, true, tok.Text + " " + tokens[i+1].Text
				}
			}
		}
	}
	return forward, reverse, cue
}

func absInt(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// Tokenize splits s into domain-aware tokens preserving byte offsets [Start, End).
// It keeps internal hyphens in compound identifiers/regions (e.g. "Hauts-de-France",
// "checkout-api", "warehouse-eu-1"), attaches '#' to numeric identifiers ("#42"),
// attaches trailing '%' or units to numbers ("38.5%", "250ms"), splits French/Italian
// elisions ("d'Occitanie" -> "d'", "Occitanie"), and segments CJK runs at particles
// and known domain vocabulary boundaries.
func Tokenize(s string) []Token {
	var out []Token
	i := 0
	n := len(s)
	for i < n {
		r, size := utf8.DecodeRuneInString(s[i:])
		if unicode.IsSpace(r) || unicode.IsControl(r) {
			i += size
			continue
		}

		// CJK script segmentation
		if isCJKRune(r) {
			j := advanceCJKToken(s, i)
			raw := s[i:j]
			out = append(out, Token{
				Text:   raw,
				Folded: raw,
				Start:  i,
				End:    j,
				Shape:  "cjk",
			})
			i = j
			continue
		}

		// Hash identifier like "#42" or "#108"
		if r == '#' && i+size < n {
			r2, _ := utf8.DecodeRuneInString(s[i+size:])
			if unicode.IsDigit(r2) || unicode.IsLetter(r2) {
				j := i + size
				for j < n {
					rj, sj := utf8.DecodeRuneInString(s[j:])
					if !unicode.IsDigit(rj) && !unicode.IsLetter(rj) && rj != '-' {
						break
					}
					j += sj
				}
				raw := s[i:j]
				out = append(out, Token{
					Text:   raw,
					Folded: foldString(raw),
					Start:  i,
					End:    j,
					Shape:  "id",
				})
				i = j
				continue
			}
		}

		// Signed or unsigned number, optionally with attached unit/percent ("38.5%", "250ms", "-4.2")
		if unicode.IsDigit(r) || ((r == '-' || r == '+') && i+size < n && isNextDigit(s[i+size:])) {
			j := i + size
			for j < n {
				rj, sj := utf8.DecodeRuneInString(s[j:])
				if unicode.IsDigit(rj) {
					j += sj
					continue
				}
				if (rj == '.' || rj == ',') && j+sj < n && isNextDigit(s[j+sj:]) {
					j += sj
					continue
				}
				break
			}
			// Check for directly attached unit/percent like "%", "ms", "€"
			for j < n {
				rj, sj := utf8.DecodeRuneInString(s[j:])
				if rj == '%' || rj == '€' || rj == '$' || rj == '£' || rj == '¥' || unicode.IsLetter(rj) {
					j += sj
					continue
				}
				break
			}
			raw := s[i:j]
			out = append(out, Token{
				Text:   raw,
				Folded: foldString(raw),
				Start:  i,
				End:    j,
				Shape:  "digit",
			})
			i = j
			continue
		}

		// Letter run (with internal hyphens/underscores when flanked by letters/digits)
		if unicode.IsLetter(r) || unicode.IsMark(r) {
			j := i + size
			for j < n {
				rj, sj := utf8.DecodeRuneInString(s[j:])
				if isCJKRune(rj) {
					break
				}
				if unicode.IsLetter(rj) || unicode.IsMark(rj) || unicode.IsDigit(rj) {
					j += sj
					continue
				}
				// Handle apostrophe in fixed expressions like "aujourd'hui", "chiffre d'affaires", "taux d'erreur"
				// vs elision prefixes like "l'entrepôt", "d'Occitanie".
				if (rj == '\'' || rj == '’') && j+sj < n {
					prefixFold := foldString(s[i:j])
					if prefixFold == "aujourd" {
						j += sj
						continue
					}
					// Split after the apostrophe for 1-2 letter elision prefixes (l', d', qu', c', n', s')
					if len(prefixFold) <= 2 {
						j += sj
						break
					}
				}
				if (rj == '-' || rj == '_') && j+sj < n {
					rNext, _ := utf8.DecodeRuneInString(s[j+sj:])
					if unicode.IsLetter(rNext) || unicode.IsDigit(rNext) {
						j += sj
						continue
					}
				}
				break
			}
			raw := s[i:j]
			trimmedRaw := strings.TrimRight(raw, "'’")
			out = append(out, Token{
				Text:   raw,
				Folded: foldString(trimmedRaw),
				Start:  i,
				End:    j,
				Shape:  tokenShape(raw),
			})
			i = j
			continue
		}

		// Punctuation or Symbol (e.g. "€", "$", "%", "?")
		raw := s[i : i+size]
		shape := "punct"
		if unicode.IsSymbol(r) || r == '€' || r == '$' || r == '£' || r == '¥' || r == '%' {
			shape = "symbol"
		}
		out = append(out, Token{
			Text:   raw,
			Folded: strings.ToLower(raw),
			Start:  i,
			End:    i + size,
			Shape:  shape,
		})
		i += size
	}
	return out
}

func isNextDigit(s string) bool {
	r, _ := utf8.DecodeRuneInString(s)
	return unicode.IsDigit(r)
}

func isCJKRune(r rune) bool {
	return unicode.Is(unicode.Han, r) || unicode.Is(unicode.Hiragana, r) || unicode.Is(unicode.Katakana, r)
}

var cjkKnownTerms = []string{
	"過去30日間", "過去7日間", "过去30天", "过去7天",
	"東京店舗", "大阪倉庫", "名古屋工場", "上海仓库", "北京门店", "深圳工厂",
	"東洋物流", "日本サプライ", "华东供应链", "远东物流",
	"純利益率", "平均客単価", "リードタイム", "在庫回転率", "库存周转率", "交货周期",
	"サプライヤー", "ベンダー", "合作伙伴", "供应商延迟",
	"粗利率", "欠品率", "解約率", "エラー率", "売上高", "納入遅延",
	"毛利率", "净利率", "缺货率", "流失率", "错误率", "营业额", "客单价",
	"今四半期", "本季度", "上季度", "供应商", "仕入先", "取引先", "承运商",
	"今週", "先週", "今月", "先月", "本周", "上周", "本月", "上月",
	"関東", "関西", "中部", "九州", "华东", "华南", "华北", "华中",
	"店舗", "倉庫", "工場", "拠点", "门店", "仓库", "工厂", "站点",
	"なぜ", "为什么", "為什麼", "影響", "影响",
}

var cjkParticles = map[rune]bool{
	'の': true, 'と': true, 'は': true, 'が': true, 'を': true, 'に': true, 'で': true, 'や': true, 'も': true,
	'的': true, '和': true, '与': true, '在': true, '比': true, '对': true, '及': true, '或': true, '？': true, '。': true, '、': true, '，': true,
}

func advanceCJKToken(s string, start int) int {
	rest := s[start:]
	for _, term := range cjkKnownTerms {
		if strings.HasPrefix(rest, term) {
			return start + len(term)
		}
	}
	r, size := utf8.DecodeRuneInString(rest)
	if cjkParticles[r] {
		return start + size
	}
	j := start + size
	for j < len(s) {
		rj, sj := utf8.DecodeRuneInString(s[j:])
		if !isCJKRune(rj) || cjkParticles[rj] {
			break
		}
		matchedKnown := false
		for _, term := range cjkKnownTerms {
			if strings.HasPrefix(s[j:], term) {
				matchedKnown = true
				break
			}
		}
		if matchedKnown {
			break
		}
		j += sj
	}
	return j
}

func tokenShape(raw string) string {
	if strings.Contains(raw, "-") || strings.Contains(raw, "_") {
		// Check if capitalized compound like "Hauts-de-France" vs lower service ID "checkout-api"
		firstRune, _ := utf8.DecodeRuneInString(raw)
		if unicode.IsUpper(firstRune) && !looksLikeServiceIdentifier(foldString(raw)) {
			return "title"
		}
		return "id"
	}
	var letters, upper int
	runes := []rune(raw)
	for _, r := range runes {
		if unicode.IsLetter(r) {
			letters++
			if unicode.IsUpper(r) {
				upper++
			}
		}
	}
	switch {
	case letters == 0:
		return "other"
	case upper == letters:
		return "upper"
	case upper == 0:
		return "lower"
	case upper == 1 && unicode.IsUpper(runes[0]):
		return "title"
	default:
		return "mixed"
	}
}

var latinFoldMap = func() map[rune]string {
	m := map[rune]string{'ß': "ss", 'æ': "ae", 'œ': "oe", 'ø': "o", 'ł': "l", 'đ': "d", 'ı': "i"}
	groups := map[string]string{
		"a": "àáâãäåāăą", "c": "çćĉċč", "d": "ď", "e": "èéêëēĕėęě", "g": "ĝğġģ", "h": "ĥ", "i": "ìíîïĩīĭįİ",
		"j": "ĵ", "k": "ķ", "l": "ĺļľ", "n": "ñńņňŉ", "o": "òóôõöōŏő", "r": "ŕŗř", "s": "śŝşšș", "t": "ţťț",
		"u": "ùúûüũūŭůűų", "w": "ŵ", "y": "ýÿŷ", "z": "źżž",
	}
	for base, chars := range groups {
		for _, r := range chars {
			m[r] = base
		}
	}
	return m
}()

func foldString(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range strings.ToLower(s) {
		if mapped, ok := latinFoldMap[r]; ok {
			b.WriteString(mapped)
			continue
		}
		b.WriteRune(r)
	}
	return strings.TrimSpace(b.String())
}

func foldTokenSlice(s string) []string {
	toks := Tokenize(s)
	out := make([]string, 0, len(toks))
	for _, t := range toks {
		if t.Folded != "" {
			out = append(out, t.Folded)
		}
	}
	return out
}

func hashTokenFeature(parts ...string) uint32 {
	h := fnv.New32a()
	for i, p := range parts {
		if i > 0 {
			_, _ = h.Write([]byte{0})
		}
		_, _ = h.Write([]byte(p))
	}
	return h.Sum32() & featureMask
}

func extractTokenFeatures(tokens []Token) [][]uint32 {
	n := len(tokens)
	out := make([][]uint32, n)
	buf := make([]uint32, 0, 16)
	for i, tok := range tokens {
		buf = buf[:0]
		buf = append(buf,
			hashTokenFeature("tok", tok.Folded),
			hashTokenFeature("shape", tok.Shape),
		)
		if len(tok.Folded) >= 3 {
			buf = append(buf,
				hashTokenFeature("pre3", tok.Folded[:3]),
				hashTokenFeature("suf3", tok.Folded[len(tok.Folded)-3:]),
			)
		}
		if i > 0 {
			buf = append(buf,
				hashTokenFeature("prev", tokens[i-1].Folded),
				hashTokenFeature("prev_shape", tokens[i-1].Shape),
			)
		}
		if i+1 < n {
			buf = append(buf,
				hashTokenFeature("next", tokens[i+1].Folded),
				hashTokenFeature("next_shape", tokens[i+1].Shape),
			)
		}
		slices.Sort(buf)
		buf = slices.Compact(buf)
		cp := make([]uint32, len(buf))
		copy(cp, buf)
		out[i] = cp
	}
	return out
}

func builtinSeedCorpus() []TrainingExample {
	return []TrainingExample{
		{
			Text: "Pourquoi le taux de rupture au magasin de Lille chez fournisseur BoisNord impacte la marge brute en Hauts-de-France ?",
			Spans: []GoldSpan{
				{Category: CategoryMetric, Text: "taux de rupture"},
				{Category: CategoryScopeSite, Text: "magasin de Lille"},
				{Category: CategorySupplierOrg, Text: "fournisseur BoisNord"},
				{Category: CategoryMetric, Text: "marge brute"},
				{Category: CategoryScopeRegion, Text: "Hauts-de-France"},
			},
		},
		{
			Text: "Evolution du panier moyen en Occitanie par rapport à Bretagne cette semaine à l'entrepôt Lyon",
			Spans: []GoldSpan{
				{Category: CategoryMetric, Text: "panier moyen"},
				{Category: CategoryScopeRegion, Text: "Occitanie"},
				{Category: CategoryComparisonTarget, Text: "par rapport à Bretagne"},
				{Category: CategoryTimeWindow, Text: "cette semaine"},
				{Category: CategoryScopeSite, Text: "entrepôt Lyon"},
			},
		},
		{
			Text: "Why did gross margin drop by 38.5% at store #42 and warehouse-eu-1 in EMEA over last 30 days vs Q2 due to supplier AcmeCorp and payment-service ?",
			Spans: []GoldSpan{
				{Category: CategoryMetric, Text: "gross margin"},
				{Category: CategoryNumberValue, Text: "38.5%"},
				{Category: CategoryScopeSite, Text: "store #42"},
				{Category: CategoryScopeSite, Text: "warehouse-eu-1"},
				{Category: CategoryScopeRegion, Text: "EMEA"},
				{Category: CategoryTimeWindow, Text: "last 30 days"},
				{Category: CategoryComparisonTarget, Text: "vs Q2"},
				{Category: CategorySupplierOrg, Text: "supplier AcmeCorp"},
				{Category: CategoryServiceSystem, Text: "payment-service"},
			},
		},
		{
			Text: "Check stockout rate and latency above 250ms on checkout-api and auth-gateway in North America during Q3 2026 for vendor #108",
			Spans: []GoldSpan{
				{Category: CategoryMetric, Text: "stockout rate"},
				{Category: CategoryMetric, Text: "latency"},
				{Category: CategoryNumberValue, Text: "250ms"},
				{Category: CategoryServiceSystem, Text: "checkout-api"},
				{Category: CategoryServiceSystem, Text: "auth-gateway"},
				{Category: CategoryScopeRegion, Text: "North America"},
				{Category: CategoryTimeWindow, Text: "Q3 2026"},
				{Category: CategorySupplierOrg, Text: "vendor #108"},
			},
		},
		{
			Text: "Impact du retard fournisseur chez partenaire LogisFrance sur le taux de rupture au site de Toulouse depuis janvier pour 10 000 € sur SAP-ERP et wms-core",
			Spans: []GoldSpan{
				{Category: CategoryMetric, Text: "retard fournisseur"},
				{Category: CategorySupplierOrg, Text: "partenaire LogisFrance"},
				{Category: CategoryMetric, Text: "taux de rupture"},
				{Category: CategoryScopeSite, Text: "site de Toulouse"},
				{Category: CategoryTimeWindow, Text: "depuis janvier"},
				{Category: CategoryNumberValue, Text: "10 000 €"},
				{Category: CategoryServiceSystem, Text: "SAP-ERP"},
				{Category: CategoryServiceSystem, Text: "wms-core"},
			},
		},
	}
}
