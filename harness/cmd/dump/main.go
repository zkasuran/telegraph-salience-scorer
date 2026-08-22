// Command dump loads a scoring module the way the node does and writes the raw
// rank_answer score for every triple in a JSON file. One dump per binary is enough
// to evaluate any monotone post-transform of the score offline, which is what the
// threshold sweeps need: the wasm is 24 MB and loading it costs seconds, while a
// sweep over thresholds costs nothing once the scores are on disk.
//
// usage: dump <module.wasm> <triples.json> <out.json>
//   triples.json: [{"id":"...","q":"...","gt":"...","a":"..."}, ...]
//   out.json:     {"module":"...","scores":{"id":0.1234,...},"order":["id",...]}
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/tetratelabs/wazero"
	"github.com/tetratelabs/wazero/api"
)

type triple struct {
	ID string `json:"id"`
	Q  string `json:"q"`
	GT string `json:"gt"`
	A  string `json:"a"`
}

func main() {
	if len(os.Args) != 4 {
		fmt.Println("usage: dump <module.wasm> <triples.json> <out.json>")
		os.Exit(2)
	}
	ctx := context.Background()
	blob, err := os.ReadFile(os.Args[2])
	if err != nil {
		fmt.Println(err)
		os.Exit(2)
	}
	var ts []triple
	if err := json.Unmarshal(blob, &ts); err != nil {
		fmt.Println(err)
		os.Exit(2)
	}
	wasm, err := os.ReadFile(os.Args[1])
	if err != nil {
		fmt.Println(err)
		os.Exit(2)
	}
	rt := wazero.NewRuntime(ctx)
	defer rt.Close(ctx)
	inst, err := rt.Instantiate(ctx, wasm)
	if err != nil {
		fmt.Println("instantiate:", err)
		os.Exit(1)
	}
	alloc, rank, mem := inst.ExportedFunction("alloc"), inst.ExportedFunction("rank_answer"), inst.Memory()
	if alloc == nil || rank == nil || mem == nil {
		fmt.Println("missing export")
		os.Exit(1)
	}
	put := func(s string) (uint32, uint32) {
		if len(s) == 0 {
			return 0, 0
		}
		r, err := alloc.Call(ctx, uint64(len(s)))
		if err != nil {
			fmt.Println("alloc:", err)
			os.Exit(1)
		}
		p := uint32(r[0])
		if !mem.Write(p, []byte(s)) {
			fmt.Println("write outside memory")
			os.Exit(1)
		}
		return p, uint32(len(s))
	}
	scores := map[string]float64{}
	order := make([]string, 0, len(ts))
	for _, t := range ts {
		qp, ql := put(t.Q)
		gp, gl := put(t.GT)
		ap, al := put(t.A)
		res, err := rank.Call(ctx, uint64(qp), uint64(ql), uint64(gp), uint64(gl), uint64(ap), uint64(al))
		if err != nil {
			fmt.Println("rank_answer:", t.ID, err)
			os.Exit(1)
		}
		scores[t.ID] = float64(api.DecodeF32(res[0]))
		order = append(order, t.ID)
	}
	out, _ := json.Marshal(map[string]any{
		"module": filepath.Base(os.Args[1]), "scores": scores, "order": order,
	})
	if err := os.WriteFile(os.Args[3], out, 0o644); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	fmt.Printf("dumped %d scores from %s to %s\n", len(scores), filepath.Base(os.Args[1]), os.Args[3])
}
