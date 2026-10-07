package train

import (
	"fmt"
	"testing"
)

func TestDevelopmentCopiesBlockedDespiteNewStoryIDsAndChangedGold(t *testing.T) {
	byRole := map[string]string{}
	for i := 0; len(byRole) < 2; i++ {
		id := fmt.Sprintf("copy-%d", i)
		byRole[DevHalf(id)] = id
	}
	d := &Dataset{}
	for _, role := range []string{RoleDevCal, RoleDevConf} {
		d.Examples = append(d.Examples, Example{Instance: "R", Task: "r_test", Split: SplitDev, StoryID: byRole[role], Input: map[string]any{"question": "What is the type?", "evidence": []any{map[string]any{"id": "source", "text": "Source text"}}}, Options: []Option{{Key: "a"}, {Key: "b"}}})
	}
	d.Examples[1].Label = 1
	d.Examples[1].Input["evidence"].([]any)[0].(map[string]any)["id"] = "renamed-source"
	r, e := CalibrationOverlap(d)
	if e != nil || r["r_test"] != 1 {
		t.Fatal(r, e)
	}
	d.Examples[1].Input["question"] = "A different scoped question?"
	r, e = CalibrationOverlap(d)
	if e != nil || len(r) != 0 {
		t.Fatal("different questions falsely merged", r, e)
	}
	d.Examples[1].Input = d.Examples[0].Input
	d.Examples[1].Split = SplitLockedTest
	r, e = CalibrationOverlap(d)
	if e != nil || len(r) != 0 {
		t.Fatal("sealed row entered comparison", r, e)
	}
}
