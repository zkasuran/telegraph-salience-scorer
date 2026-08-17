// Command harness runs a Telegraph scoring module through the gates the node
// applies before it will let a module score real traffic, plus a gaming suite of
// our own. It loads the .wasm exactly the way the node does (wazero, no host
// imports, strings written through the module's own alloc), so a module that
// passes here is being measured under the same rules.
//
// usage: harness <benchmark.json> <attacks.json> <candidate.wasm> [baseline.wasm ...]
//
// The first module is the candidate; any others are baselines to beat.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/tetratelabs/wazero"
	"github.com/tetratelabs/wazero/api"
)

type benchCase struct {
	ID     string `json:"id"`
	Intent string `json:"intent"`
	Q      string `json:"question"`
	GT     string `json:"ground_truth"`
	Good   string `json:"good"`
	Bad    string `json:"bad"`
}

type attackCase struct {
	Name   string `json:"name"`
	Rule   string `json:"rule"`
	Q      string `json:"question"`
	GT     string `json:"ground_truth"`
	Honest string `json:"honest"`
	Attack string `json:"attack"`
	Why    string `json:"why"`
}

type mod struct {
	name  string
	rt    wazero.Runtime
	inst  api.Module
	alloc api.Function
	rank  api.Function
	mem   api.Memory
}

func loadMod(ctx context.Context, path string) (*mod, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	rt := wazero.NewRuntime(ctx)
	inst, err := rt.Instantiate(ctx, b)
	if err != nil {
		// No host module is registered on purpose: a WASI build fails right here,
		// the same way it fails on the node.
		return nil, fmt.Errorf("instantiate: %w", err)
	}
	m := &mod{
		name:  filepath.Base(path),
		rt:    rt,
		inst:  inst,
		alloc: inst.ExportedFunction("alloc"),
		rank:  inst.ExportedFunction("rank_answer"),
		mem:   inst.Memory(),
	}
	if m.alloc == nil || m.rank == nil || inst.ExportedFunction("dealloc") == nil || m.mem == nil {
		return nil, fmt.Errorf("missing export: alloc, dealloc, rank_answer and linear memory are all required")
	}
	return m, nil
}

func (m *mod) put(ctx context.Context, s string) (uint32, uint32, error) {
	if len(s) == 0 {
		return 0, 0, nil
	}
	r, err := m.alloc.Call(ctx, uint64(len(s)))
	if err != nil {
		return 0, 0, fmt.Errorf("alloc: %w", err)
	}
	p := uint32(r[0])
	if !m.mem.Write(p, []byte(s)) {
		return 0, 0, fmt.Errorf("writing %d bytes at %d fell outside module memory", len(s), p)
	}
	return p, uint32(len(s)), nil
}

func (m *mod) score(ctx context.Context, q, gt, ans string) (float64, error) {
	qp, ql, err := m.put(ctx, q)
	if err != nil {
		return 0, err
	}
	gp, gl, err := m.put(ctx, gt)
	if err != nil {
		return 0, err
	}
	ap, al, err := m.put(ctx, ans)
	if err != nil {
		return 0, err
	}
	res, err := m.rank.Call(ctx, uint64(qp), uint64(ql), uint64(gp), uint64(gl), uint64(ap), uint64(al))
	if err != nil {
		return 0, fmt.Errorf("rank_answer: %w", err)
	}
	v := float64(api.DecodeF32(res[0]))
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return 0, fmt.Errorf("rank_answer returned %v", v)
	}
	return v, nil
}

type check struct {
	Name   string `json:"name"`
	OK     bool   `json:"ok"`
	Detail string `json:"detail"`
}

// stage1 mirrors the node's structural gates: the module loads and exports what
// it must, a blank answer is exactly zero, a perfect answer scores high, a
// correct answer beats an unrelated one and awkward input neither traps nor
// escapes [0,1].
func stage1(ctx context.Context, m *mod, cases []benchCase) []check {
	var out []check
	add := func(name string, ok bool, format string, args ...any) {
		out = append(out, check{name, ok, fmt.Sprintf(format, args...)})
	}
	c0 := cases[0]

	s, err := m.score(ctx, c0.Q, c0.GT, "")
	add("empty answer scores exactly 0", err == nil && s == 0, "score=%.4f err=%v", s, err)
	s, err = m.score(ctx, c0.Q, c0.GT, " \t\n \r ")
	add("whitespace-only answer scores exactly 0", err == nil && s == 0, "score=%.4f err=%v", s, err)

	worstSelf, worstGap, gapCase := 1.0, 1.0, ""
	failed := false
	for i, c := range cases {
		self, e1 := m.score(ctx, c.Q, c.GT, c.GT)
		cross, e2 := m.score(ctx, c.Q, c.GT, cases[(i+1)%len(cases)].GT)
		if e1 != nil || e2 != nil {
			add("self/cross scoring for "+c.ID, false, "errors: %v / %v", e1, e2)
			failed = true
			continue
		}
		if self < worstSelf {
			worstSelf = self
		}
		if self-cross < worstGap {
			worstGap, gapCase = self-cross, c.ID
		}
	}
	add("perfect answer scores >= 0.75 everywhere", !failed && worstSelf >= 0.75, "worst_self_match=%.4f", worstSelf)
	add("self-match beats unrelated cross-match", !failed && worstGap > 0, "narrowest gap %.4f (%s)", worstGap, gapCase)

	long := strings.Repeat("lorem ipsum dolor sit amet consectetur adipiscing ", 1600)
	s, err = m.score(ctx, c0.Q, c0.GT, long)
	add(fmt.Sprintf("%dKB answer does not trap", len(long)/1024), err == nil && s >= 0 && s <= 1, "score=%.4f err=%v", s, err)
	s, err = m.score(ctx, c0.Q, long, long)
	add("oversized ground truth does not trap", err == nil && s >= 0 && s <= 1, "score=%.4f err=%v", s, err)

	weird := "🚀🌙 登月成功了 مرحبا بالعالم Привет мир ✅ \x00\xff\xfe binary \x01"
	s, err = m.score(ctx, c0.Q, c0.GT, weird)
	add("emoji/CJK/RTL/invalid-UTF8 answer does not trap", err == nil && s >= 0 && s <= 1, "score=%.4f err=%v", s, err)
	s, err = m.score(ctx, weird, weird, weird)
	add("non-text ground truth self-matches", err == nil && s >= 0.75, "score=%.4f err=%v", s, err)
	return out
}

// metrics are the same numbers the node records on a registration, so a local
// run can be compared with what the node reports back.
type metrics struct {
	Module          string                `json:"module"`
	ComparableCases int                   `json:"comparable_cases"`
	Wins            int                   `json:"wins"`
	Ties            int                   `json:"ties"`
	Losses          []string              `json:"losses"`
	CandidateMargin float64               `json:"candidate_margin"`
	MeanGood        float64               `json:"mean_good"`
	MeanBad         float64               `json:"mean_bad"`
	WorstSelfMatch  float64               `json:"worst_self_match"`
	ScoreStddev     float64               `json:"score_stddev"`
	MinCaseMargin   float64               `json:"min_case_margin"`
	PerCase         map[string][2]float64 `json:"per_case"`
}

func stage2(ctx context.Context, m *mod, cases []benchCase) (metrics, error) {
	res := metrics{Module: m.name, WorstSelfMatch: 1, MinCaseMargin: 1, PerCase: map[string][2]float64{}}
	var all []float64
	var sumGood, sumBad, sumMargin float64
	for _, c := range cases {
		good, err := m.score(ctx, c.Q, c.GT, c.Good)
		if err != nil {
			return res, fmt.Errorf("%s good: %w", c.ID, err)
		}
		bad, err := m.score(ctx, c.Q, c.GT, c.Bad)
		if err != nil {
			return res, fmt.Errorf("%s bad: %w", c.ID, err)
		}
		self, err := m.score(ctx, c.Q, c.GT, c.GT)
		if err != nil {
			return res, fmt.Errorf("%s self: %w", c.ID, err)
		}
		res.PerCase[c.ID] = [2]float64{good, bad}
		if self < res.WorstSelfMatch {
			res.WorstSelfMatch = self
		}
		if good-bad < res.MinCaseMargin {
			res.MinCaseMargin = good - bad
		}
		switch {
		case good > bad:
			res.Wins++
		case good == bad:
			res.Ties++
			res.Losses = append(res.Losses, c.ID+"(tie)")
		default:
			res.Losses = append(res.Losses, c.ID)
		}
		sumGood, sumBad, sumMargin = sumGood+good, sumBad+bad, sumMargin+good-bad
		all = append(all, good, bad)
	}
	n := float64(len(cases))
	res.ComparableCases = len(cases)
	res.CandidateMargin, res.MeanGood, res.MeanBad = sumMargin/n, sumGood/n, sumBad/n
	res.ScoreStddev = stddev(all)
	return res, nil
}

func stddev(v []float64) float64 {
	if len(v) < 2 {
		return 0
	}
	var mean float64
	for _, x := range v {
		mean += x
	}
	mean /= float64(len(v))
	var s float64
	for _, x := range v {
		s += (x - mean) * (x - mean)
	}
	return math.Sqrt(s / float64(len(v)-1))
}

// runAttacks checks the module cannot be talked into a good score without being
// right and that ordinary noise does not destroy a right answer.
func runAttacks(ctx context.Context, m *mod, cases []attackCase) []check {
	var out []check
	for _, a := range cases {
		honest, e1 := m.score(ctx, a.Q, a.GT, a.Honest)
		attack, e2 := m.score(ctx, a.Q, a.GT, a.Attack)
		if e1 != nil || e2 != nil {
			out = append(out, check{a.Name, false, fmt.Sprintf("errors: %v / %v", e1, e2)})
			continue
		}
		var ok bool
		switch a.Rule {
		case "below_honest":
			ok = attack < honest-0.10
		case "near_zero":
			ok = attack <= 0.20
		case "near_honest":
			ok = attack >= honest-0.15
		default:
			out = append(out, check{a.Name, false, "unknown rule " + a.Rule})
			continue
		}
		out = append(out, check{
			fmt.Sprintf("%s [%s]", a.Name, a.Rule),
			ok,
			fmt.Sprintf("honest=%.4f attack=%.4f", honest, attack),
		})
	}
	return out
}

func mark(ok bool) string {
	if ok {
		return "[ok]  "
	}
	return "[FAIL]"
}

func mustJSON(path string, v any) {
	b, err := os.ReadFile(path)
	if err != nil {
		fmt.Println("cannot read", path, err)
		os.Exit(2)
	}
	if err := json.Unmarshal(b, v); err != nil {
		fmt.Println("cannot parse", path, err)
		os.Exit(2)
	}
}

type report struct {
	Metrics metrics `json:"metrics"`
	Stage1  []check `json:"stage1"`
	Attacks []check `json:"attacks"`
}

func main() {
	if len(os.Args) < 4 {
		fmt.Println("usage: harness <benchmark.json> <attacks.json> <candidate.wasm> [baseline.wasm ...]")
		os.Exit(2)
	}
	ctx := context.Background()

	var bench struct {
		Cases []benchCase `json:"cases"`
	}
	var atk struct {
		Cases []attackCase `json:"cases"`
	}
	mustJSON(os.Args[1], &bench)
	mustJSON(os.Args[2], &atk)
	if len(bench.Cases) == 0 {
		fmt.Println("benchmark has no cases")
		os.Exit(2)
	}

	reports := map[string]report{}
	var order []string
	failures := 0

	for i, path := range os.Args[3:] {
		m, err := loadMod(ctx, path)
		if err != nil {
			fmt.Printf("\n=== %s ===\n  [FAIL] load: %v\n", filepath.Base(path), err)
			failures++
			continue
		}
		defer m.rt.Close(ctx)

		s1 := stage1(ctx, m, bench.Cases)
		mx, err := stage2(ctx, m, bench.Cases)
		if err != nil {
			fmt.Printf("\n=== %s ===\n  [FAIL] benchmark: %v\n", m.name, err)
			failures++
			continue
		}
		ats := runAttacks(ctx, m, atk.Cases)
		reports[m.name] = report{Metrics: mx, Stage1: s1, Attacks: ats}
		order = append(order, m.name)

		role := "baseline"
		if i == 0 {
			role = "CANDIDATE"
		}
		fmt.Printf("\n=== %s (%s) ===\n", m.name, role)
		for _, c := range s1 {
			fmt.Printf("  %s %-46s %s\n", mark(c.OK), c.Name, c.Detail)
			if !c.OK && i == 0 {
				failures++
			}
		}
		fmt.Printf("  -- benchmark, %d cases --\n", mx.ComparableCases)
		fmt.Printf("     candidate_margin %.4f | wins %d/%d | ties %d | mean good %.4f bad %.4f\n",
			mx.CandidateMargin, mx.Wins, mx.ComparableCases, mx.Ties, mx.MeanGood, mx.MeanBad)
		fmt.Printf("     worst_self_match %.4f | score_stddev %.4f | worst case margin %.4f\n",
			mx.WorstSelfMatch, mx.ScoreStddev, mx.MinCaseMargin)
		if len(mx.Losses) > 0 {
			fmt.Printf("     not won: %s\n", strings.Join(mx.Losses, ", "))
		}
		fmt.Printf("  -- gaming and robustness suite --\n")
		for _, c := range ats {
			fmt.Printf("  %s %-46s %s\n", mark(c.OK), c.Name, c.Detail)
			if !c.OK && i == 0 {
				failures++
			}
		}
	}

	if len(order) > 1 {
		cand := reports[order[0]].Metrics
		fmt.Printf("\n=== head to head ===\n")
		for _, name := range order[1:] {
			base := reports[name].Metrics
			ok := cand.CandidateMargin >= base.CandidateMargin && cand.Wins >= base.Wins
			fmt.Printf("  %s vs %-28s margin %.4f vs %.4f | wins %d vs %d\n",
				mark(ok), name, cand.CandidateMargin, base.CandidateMargin, cand.Wins, base.Wins)
			if !ok {
				failures++
			}
		}
		fmt.Printf("\n%-22s %20s %20s\n", "case (good / bad)", order[0], order[1])
		for _, c := range bench.Cases {
			a := reports[order[0]].Metrics.PerCase[c.ID]
			b := reports[order[1]].Metrics.PerCase[c.ID]
			fmt.Printf("%-22s %9.4f %9.4f %9.4f %9.4f\n", c.ID, a[0], a[1], b[0], b[1])
		}
	}

	out := os.Getenv("REPORT")
	if out == "" {
		out = "harness-report.json"
	}
	blob, err := json.MarshalIndent(reports, "", "  ")
	if err == nil {
		if err := os.WriteFile(out, blob, 0o644); err == nil {
			fmt.Printf("\nreport written to %s\n", out)
		}
	}

	if failures > 0 {
		fmt.Printf("\n%d gate(s) failed for the candidate\n", failures)
		os.Exit(1)
	}
	fmt.Printf("\nall candidate gates passed\n")
}