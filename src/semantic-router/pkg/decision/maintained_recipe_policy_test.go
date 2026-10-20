package decision

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Exercise the maintained policy itself rather than a duplicate test-only tree.
func maintainedRecipeEngine(t *testing.T, name string) *DecisionEngine {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "..", "..", "..", "config", "recipes", name, "config.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := config.ParseYAMLBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	return NewDecisionEngine(cfg.KeywordRules, cfg.EmbeddingRules, cfg.Categories, cfg.Decisions, config.RoutingStrategyPriority)
}

func TestBalanceQuestionPolicy(t *testing.T) {
	engine := maintainedRecipeEngine(t, "balance")
	lanes := []struct{ label, lane string }{
		{"legal", "premium_legal"},
		{"proof", "formal_math_proof"},
		{"argument", "reasoning_deep"},
		{"systems", "complex_specialist"},
		{"science", "complex_specialist"},
		{"code", "medium_code_general"},
		{"health", "verified_health"},
		{"explain", "medium_explainer"},
		{"creative", "medium_creative"},
		{"facts", "fast_qa"},
		{"chat", "simple_general"},
	}
	var tests []evaluateSignalsCase
	for _, l := range lanes {
		tests = append(tests, evaluateSignalsCase{
			name:             "task " + l.label + " picks " + l.lane,
			signals:          &SignalMatches{DecisionRules: []string{"task:" + l.label}},
			expectedDecision: l.lane,
		})
	}
	unknown := map[string]string{}
	for _, name := range []string{"task", "verification", "correction", "clarify"} {
		unknown["decision:"+name] = "decision_rejected"
	}
	tests = append(tests, []evaluateSignalsCase{
		{
			name:             "an explanation with a source request is verified",
			signals:          &SignalMatches{DecisionRules: []string{"task:explain", "verification"}},
			expectedDecision: "verified_explainer",
		},
		{
			name:             "a source request alone picks no lane",
			signals:          &SignalMatches{DecisionRules: []string{"verification"}},
			expectedDecision: "casual_chat",
		},
		{
			name:             "a correction needs an earlier answer",
			signals:          &SignalMatches{DecisionRules: []string{"task:code", "correction"}},
			expectedDecision: "medium_code_general",
		},
		{
			name: "a correction after an answer outranks the code lane",
			signals: &SignalMatches{
				ConversationRules: []string{"has_prior_answer"},
				DecisionRules:     []string{"task:code", "correction"},
			},
			expectedDecision: "feedback_wrong_answer_verified",
		},
		{
			name: "the legal lane outranks a correction",
			signals: &SignalMatches{
				ConversationRules: []string{"has_prior_answer"},
				DecisionRules:     []string{"task:legal", "correction"},
			},
			expectedDecision: "premium_legal",
		},
		{
			name: "a clarification after an answer outranks the explainer lane",
			signals: &SignalMatches{
				ConversationRules: []string{"has_prior_answer"},
				DecisionRules:     []string{"task:explain", "clarify"},
			},
			expectedDecision: "feedback_need_clarification",
		},
		{
			name: "a clarification that is also a correction recovers",
			signals: &SignalMatches{
				ConversationRules: []string{"has_prior_answer"},
				DecisionRules:     []string{"task:chat", "clarify", "correction"},
			},
			expectedDecision: "feedback_wrong_answer_verified",
		},
		{
			name: "an image goes to the visual lane whatever the task",
			signals: &SignalMatches{
				ConversationRules: []string{"has_image"},
				DecisionRules:     []string{"task:legal"},
			},
			expectedDecision: "omni",
		},
		{
			name: "unknown answers fall back",
			signals: &SignalMatches{
				ConversationRules: []string{"has_prior_answer"},
				SignalErrors:      unknown,
			},
			expectedDecision: "casual_chat",
		},
	}...)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := engine.EvaluateDecisionsWithSignals(tt.signals)
			assertDecisionResult(t, result, err, tt.expectedDecision)
		})
	}
}

func TestAgentTaskIntentKeepsPrivacyPriority(t *testing.T) {
	engine := maintainedRecipeEngine(t, "agent")
	tests := []evaluateSignalsCase{
		{
			name: "comparison does not escalate on learned complexity alone",
			signals: &SignalMatches{
				DomainRules:     []string{"computer science"},
				KeywordRules:    []string{"comparison_request"},
				EmbeddingRules:  []string{"research_synthesis"},
				ComplexityRules: []string{"general_reasoning:hard", "evidence_synthesis:hard"},
				ContextRules:    []string{"medium_context"},
				ProjectionRules: []string{"policy_privacy_cloud_allowed", "policy_security_standard", "balance_complex", "non_agentic"},
			},
			expectedDecision: "medium_general",
		},
		{
			name: "comparison with explicit reasoning still escalates",
			signals: &SignalMatches{
				KeywordRules:    []string{"comparison_request", "reasoning_request_markers"},
				ProjectionRules: []string{"policy_privacy_cloud_allowed", "policy_security_standard", "balance_complex"},
			},
			expectedDecision: "complex_general",
		},
		{
			name: "comparison with dense constraints still escalates",
			signals: &SignalMatches{
				KeywordRules:    []string{"comparison_request"},
				StructureRules:  []string{"constraint_dense"},
				ProjectionRules: []string{"policy_privacy_cloud_allowed", "policy_security_standard", "balance_complex"},
			},
			expectedDecision: "complex_general",
		},
		{
			name: "editing code with uncertain embedding",
			signals: &SignalMatches{
				DomainRules:     []string{"computer science"},
				KeywordRules:    []string{"code_edit_action", "code_artifact"},
				ProjectionRules: []string{"policy_privacy_cloud_allowed", "policy_security_standard", "balance_simple"},
			},
			expectedDecision: "domain_code",
		},
		{
			name: "software topic alone is not coding",
			signals: &SignalMatches{
				DomainRules:     []string{"computer science"},
				ProjectionRules: []string{"policy_privacy_cloud_allowed", "policy_security_standard", "balance_simple"},
			},
			expectedDecision: "simple_general",
		},
		{
			name: "business semantics retain commercial lane",
			signals: &SignalMatches{
				DomainRules:     []string{"computer science"},
				EmbeddingRules:  []string{"business_analysis"},
				ProjectionRules: []string{"policy_privacy_cloud_allowed", "policy_security_standard", "balance_simple"},
			},
			expectedDecision: "domain_business",
		},
		{
			name: "privacy outranks editing",
			signals: &SignalMatches{
				DomainRules:     []string{"computer science"},
				KeywordRules:    []string{"code_edit_action", "code_artifact"},
				ProjectionRules: []string{"policy_privacy_local_only", "policy_security_standard", "balance_simple"},
			},
			expectedDecision: "local_privacy_policy",
		},
		{
			name: "daily comparison is not a business request",
			signals: &SignalMatches{
				DomainRules:     []string{"business"},
				KeywordRules:    []string{"comparison_request"},
				ProjectionRules: []string{"policy_privacy_cloud_allowed", "policy_security_standard", "balance_simple"},
			},
			expectedDecision: "medium_general",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := engine.EvaluateDecisionsWithSignals(tt.signals)
			assertDecisionResult(t, result, err, tt.expectedDecision)
		})
	}
}
