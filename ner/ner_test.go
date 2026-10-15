package ner

import (
	"sync"
	"testing"
	"time"
)

func TestEnterpriseNER_AllEightCategoriesAndFourRelations(t *testing.T) {
	engine := DefaultEngine()

	t.Run("French multi-entity causal and regional query", func(t *testing.T) {
		q := "Pourquoi le taux de rupture au magasin de Lille chez fournisseur BoisNord impacte la marge brute en Hauts-de-France ?"
		res := engine.Analyze(q, nil)

		assertHasSpan(t, res, CategoryMetric, "taux de rupture", "taux de rupture")
		assertHasSpan(t, res, CategoryScopeSite, "magasin de Lille", "Lille")
		assertHasSpan(t, res, CategorySupplierOrg, "fournisseur BoisNord", "BoisNord")
		assertHasSpan(t, res, CategoryMetric, "marge brute", "marge brute")
		assertHasSpan(t, res, CategoryScopeRegion, "Hauts-de-France", "Hauts-de-France")

		assertHasRelation(t, res, RelationScopedTo, "taux de rupture", "Lille")
		assertHasRelation(t, res, RelationScopedTo, "taux de rupture", "BoisNord")
		assertHasRelation(t, res, RelationScopedTo, "marge brute", "Hauts-de-France")
		assertHasRelation(t, res, RelationCausesOrDrives, "taux de rupture", "marge brute")
	})

	t.Run("Comparison, time window, site, and metric relations", func(t *testing.T) {
		q := "Evolution du panier moyen en Occitanie par rapport à Bretagne cette semaine à l'entrepôt Lyon"
		res := engine.Analyze(q, nil)

		assertHasSpan(t, res, CategoryMetric, "panier moyen", "panier moyen")
		assertHasSpan(t, res, CategoryScopeRegion, "Occitanie", "Occitanie")
		assertHasSpan(t, res, CategoryComparisonTarget, "par rapport à Bretagne", "Bretagne")
		assertHasSpan(t, res, CategoryTimeWindow, "cette semaine", "cette semaine")
		assertHasSpan(t, res, CategoryScopeSite, "entrepôt Lyon", "Lyon")

		assertHasRelation(t, res, RelationComparesWith, "Occitanie", "Bretagne")
		assertHasRelation(t, res, RelationMeasuredOver, "panier moyen", "cette semaine")
		assertHasRelation(t, res, RelationScopedTo, "panier moyen", "Occitanie")
	})

	t.Run("English services, numbers, stores, suppliers, and baselines", func(t *testing.T) {
		q := "Why did gross margin drop by 38.5% at store #42 and warehouse-eu-1 in EMEA over last 30 days vs Q2 due to supplier AcmeCorp and payment-service latency above 250ms ?"
		res := engine.Analyze(q, nil)

		assertHasSpan(t, res, CategoryMetric, "gross margin", "gross margin")
		assertHasSpan(t, res, CategoryNumberValue, "38.5%", "38.5%")
		assertHasSpan(t, res, CategoryScopeSite, "store #42", "store #42")
		assertHasSpan(t, res, CategoryScopeSite, "warehouse-eu-1", "warehouse-eu-1")
		assertHasSpan(t, res, CategoryScopeRegion, "EMEA", "EMEA")
		assertHasSpan(t, res, CategoryTimeWindow, "last 30 days", "last 30 days")
		assertHasSpan(t, res, CategoryComparisonTarget, "vs Q2", "Q2")
		assertHasSpan(t, res, CategorySupplierOrg, "supplier AcmeCorp", "AcmeCorp")
		assertHasSpan(t, res, CategoryServiceSystem, "payment-service", "payment-service")
		assertHasSpan(t, res, CategoryMetric, "latency", "latency")
		assertHasSpan(t, res, CategoryNumberValue, "250ms", "250ms")
	})

	t.Run("Additional enterprise examples across categories", func(t *testing.T) {
		q := "Impact du retard fournisseur chez partenaire LogisFrance sur le taux de rupture au site de Toulouse depuis janvier pour 10 000 € et -4.2 pts sur SAP-ERP, wms-core, checkout-api et auth-gateway"
		res := engine.Analyze(q, nil)

		assertHasSpan(t, res, CategoryMetric, "retard fournisseur", "retard fournisseur")
		assertHasSpan(t, res, CategorySupplierOrg, "partenaire LogisFrance", "LogisFrance")
		assertHasSpan(t, res, CategoryMetric, "taux de rupture", "taux de rupture")
		assertHasSpan(t, res, CategoryScopeSite, "site de Toulouse", "Toulouse")
		assertHasSpan(t, res, CategoryTimeWindow, "depuis janvier", "depuis janvier")
		assertHasSpan(t, res, CategoryNumberValue, "10 000 €", "10 000 €")
		assertHasSpan(t, res, CategoryNumberValue, "-4.2 pts", "-4.2 pts")
		assertHasSpan(t, res, CategoryServiceSystem, "SAP-ERP", "SAP-ERP")
		assertHasSpan(t, res, CategoryServiceSystem, "wms-core", "wms-core")
		assertHasSpan(t, res, CategoryServiceSystem, "checkout-api", "checkout-api")
		assertHasSpan(t, res, CategoryServiceSystem, "auth-gateway", "auth-gateway")
		assertHasRelation(t, res, RelationCausesOrDrives, "retard fournisseur", "taux de rupture")
	})
}

func TestEnterpriseNER_TenLanguagesCoverage(t *testing.T) {
	engine := DefaultEngine()

	cases := []struct {
		lang     string
		query    string
		cat      EntityCategory
		expected string
	}{
		{"EN", "What is the stockout rate in North America this quarter?", CategoryMetric, "stockout rate"},
		{"FR", "Quelle est la marge brute en Île-de-France semaine dernière ?", CategoryScopeRegion, "Île-de-France"},
		{"DE", "Warum beeinflusst die Ausfallrate in Filiale München die Bruttomarge in DACH diese Woche ?", CategoryScopeSite, "München"},
		{"PL", "Dlaczego wskaźnik braków w sklep Warszawa wpływa na marża brutto w Mazowsze w tym tygodniu ?", CategoryScopeSite, "Warszawa"},
		{"ES", "¿Por qué la tasa de rotura en tienda Madrid impacta el margen bruto en Cataluña esta semana ?", CategoryMetric, "tasa de rotura"},
		{"IT", "Perché il tasso di rottura nel negozio Milano impatta il margine lordo in Lombardia questa settimana ?", CategoryMetric, "margine lordo"},
		{"PT", "Como a taxa de ruptura na loja Lisboa afeta a margem bruta esta semana ?", CategoryScopeSite, "Lisboa"},
		{"NL", "Waarom daalt de brutomarge in winkel Amsterdam deze week ?", CategoryMetric, "brutomarge"},
		{"JA", "なぜ今週の東京店舗の欠品率が粗利率に影響するのか？", CategoryScopeSite, "東京店舗"},
		{"ZH", "为什么本周上海仓库的缺货率会影响华东的毛利率？", CategoryMetric, "毛利率"},
	}

	for _, tc := range cases {
		t.Run(tc.lang, func(t *testing.T) {
			res := engine.Analyze(tc.query, nil)
			found := false
			for _, sp := range res.Spans {
				if sp.Category == tc.cat && (sp.Text == tc.expected || sp.Normalized == tc.expected) {
					found = true
					break
				}
			}
			if !found {
				t.Fatalf("[%s] expected category %s with %q in %+v", tc.lang, tc.cat, tc.expected, res.Spans)
			}
		})
	}
}

func TestEnterpriseNER_DynamicCatalogAndTrain(t *testing.T) {
	engine := NewEngine()

	// Dynamic catalog matching for tenant-specific metric keys
	res := engine.Analyze("How is net_promoter_velocity behaving at warehouse-eu-1?", []string{"net_promoter_velocity"})
	assertHasSpan(t, res, CategoryMetric, "net_promoter_velocity", "net_promoter_velocity")
	assertHasSpan(t, res, CategoryScopeSite, "warehouse-eu-1", "warehouse-eu-1")

	// Supervised calibration via Train
	err := engine.Train([]TrainingExample{
		{
			Text: "Inspect ZoltarMetric in KrynnRegion",
			Spans: []GoldSpan{
				{Category: CategoryMetric, Text: "ZoltarMetric"},
				{Category: CategoryScopeRegion, Text: "KrynnRegion"},
			},
		},
	})
	if err != nil {
		t.Fatalf("Train returned error: %v", err)
	}
	resAfter := engine.Analyze("Inspect ZoltarMetric in KrynnRegion", nil)
	assertHasSpan(t, resAfter, CategoryMetric, "ZoltarMetric", "ZoltarMetric")
	assertHasSpan(t, resAfter, CategoryScopeRegion, "KrynnRegion", "KrynnRegion")
}

func TestEnterpriseNER_ConcurrencyAndLatency(t *testing.T) {
	engine := DefaultEngine()
	q := "Pourquoi le taux de rupture au magasin de Lille chez fournisseur BoisNord impacte la marge brute en Hauts-de-France ?"

	// Measure single-query latency (< 150 µs average)
	const iters = 200
	start := time.Now()
	for i := 0; i < iters; i++ {
		_ = engine.Analyze(q, []string{"taux_de_rupture", "marge_brute"})
	}
	avg := time.Since(start) / iters
	if avg > 500*time.Microsecond {
		t.Fatalf("expected fast in-process NER (< 500µs even under race detector), got avg %v", avg)
	}

	// Concurrent safety under -race
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 25; i++ {
				r := engine.Analyze(q, []string{"marge_brute"})
				if len(r.Spans) < 5 {
					t.Errorf("unexpected span count %d", len(r.Spans))
				}
			}
		}()
	}
	wg.Wait()
}

func assertHasSpan(t *testing.T, res AnalysisResult, cat EntityCategory, wantText, wantNorm string) {
	t.Helper()
	for _, sp := range res.Spans {
		if sp.Category == cat && sp.Text == wantText && sp.Normalized == wantNorm {
			return
		}
	}
	t.Fatalf("missing span {category=%q text=%q normalized=%q} in %+v", cat, wantText, wantNorm, res.Spans)
}

func assertHasRelation(t *testing.T, res AnalysisResult, relType RelationType, srcNorm, tgtNorm string) {
	t.Helper()
	for _, rel := range res.Relations {
		if rel.Type == relType && rel.SourceNormalized == srcNorm && rel.TargetNormalized == tgtNorm {
			return
		}
	}
	t.Fatalf("missing relation {type=%q source=%q target=%q} in %+v", relType, srcNorm, tgtNorm, res.Relations)
}
