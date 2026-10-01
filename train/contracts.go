package train

import "slices"

// Contract names and closed vocabularies of the three model contracts this
// package reads and writes. The authoritative text is the "Model engine and
// data" section of docs/architecture/intelligence-contracts.md (converged on
// w10/laya-train and w10/laya-integrate); the values below are the ones the
// Python trainer (tools/laya-train/laya_train/contracts.py), the Go loader
// (laya.LoadArtifact) and Hive's routermodel check. A value outside a closed
// set is a contract violation and is refused, never coerced.
const (
	// DatasetContract is the JSONL training and evaluation example contract.
	DatasetContract = "hive.laya-dataset/v1"
	// ArtifactContract is a trained head directory.
	ArtifactContract = "hive.laya-artifact/v1"
	// PromptFormat is how an example becomes a Laya state
	// (laya.RenderPromptState).
	PromptFormat = "hive.laya-prompt/v1"
	// SequenceFormat is Laya 0.3.5's build_sequence layout
	// (laya.BuildSequence).
	SequenceFormat = "laya-build-sequence/0.3.5"
	// LLMInteractionContract names the provenance record of model-written
	// text.
	LLMInteractionContract = "hive.llm-interaction/v1"
	// EvidenceTier is the tier of every head trained on generated data.
	EvidenceTier = "synthetic_engineering"
	// TrainerVersion identifies this trainer in manifests.
	TrainerVersion = "laya-go-train/2"
)

// Dataset splits.
const (
	SplitTrain           = "train"
	SplitDev             = "dev"
	SplitHeldoutDomain   = "heldout_domain"
	SplitHeldoutLanguage = "heldout_language"
	SplitLockedTest      = "locked_test"
	// RoleDevCal and RoleDevConf are the two halves dev is divided into by
	// story (DevHalf): dev_cal early-stops the scorer and fits the
	// temperatures and the act head; dev_conf is used only for the act
	// thresholds, so their certificate is computed on data nothing was
	// fitted on.
	RoleDevCal  = "dev_cal"
	RoleDevConf = "dev_conf"
)

// Splits are the contract's split values.
var Splits = []string{SplitTrain, SplitDev, SplitHeldoutDomain, SplitHeldoutLanguage, SplitLockedTest}

// fitSplits are the splits a model is fitted on; held-out and locked splits
// share no story, held-out domain or held-out language with them.
var fitSplits = []string{SplitTrain, SplitDev}

// Instances.
const (
	InstanceQ = "Q"
	InstanceR = "R"
	InstanceA = "A"
)

// Instances are the contract's instance values: Q reads a question against
// a catalog, R judges evidence, A audits a drafted answer.
var Instances = []string{InstanceQ, InstanceR, InstanceA}

// LabelSources are the contract's label_source values.
var LabelSources = []string{"spec", "template", "llm_paraphrase", "llm_translation", "claude_author"}

// llmLabelSources wrote text with a model and must carry the
// hive.llm-interaction/v1 id of every call.
var llmLabelSources = []string{"llm_paraphrase", "llm_translation", "claude_author"}

// ActOutputs are the act head's outputs; index 0 is act
// (laya.Answer.ActProbability).
var ActOutputs = []string{"act", "escalate"}

// ProbabilityQuantum is the grid scores are rounded to before any threshold
// comparison, as QuantizeProbability does.
const ProbabilityQuantum = 1e-4

// DefaultInstructions are the instructions of each known task, identical to
// the Python trainer's (prompt.py DEFAULT_INSTRUCTIONS). They are English on
// purpose: the instruction names the task for the head, the user's language
// lives in the state. A trained artifact records the instructions of every
// task, and the runtime asks with those.
var DefaultInstructions = map[string]string{
	"q_operation":          "Which operation does the business question ask for?",
	"q_metric_choice":      "Which catalog metric does the question ask about?",
	"q_scope_choice":       "Which scope does the question restrict the metric to?",
	"q_time_supported":     "Can the time window of the question be answered with a supported calendar window?",
	"q_single_series":      "Does the question ask about exactly one metric series?",
	"q_act_or_escalate":    "Can the question be answered directly, without clarification or escalation?",
	"r_evidence_relevance": "How relevant is the evidence item to the question?",
	"r_answer_rank":        "Which evidence item best answers the question?",
	"a_causal_licensed":    "Is the causal wording of the sentence licensed by the grounded draft?",
}

// DefaultInstruction returns a task's default instructions: the known text,
// else "Answer the <task words> task for this input.", as prompt.py does.
func DefaultInstruction(task string) string {
	if s, ok := DefaultInstructions[task]; ok {
		return s
	}
	words := []byte(task)
	for i, c := range words {
		if c == '_' {
			words[i] = ' '
		}
	}
	return "Answer the " + string(words) + " task for this input."
}

func isOneOf(v string, set []string) bool { return slices.Contains(set, v) }
