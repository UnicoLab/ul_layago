package tensor

import (
	"go/ast"
	"go/build"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// kernelFieldTypes are the microkernel fields that hold assembly functions.
var kernelFieldTypes = []string{"f32", "q8"}

// TestEveryAssemblyCallIsAPreemptionPoint is the default-suite guard for
// the property TestAssemblyKernelsStopPromptlyForTheRuntime times (opt-in,
// wall clock): an assembly GEMM kernel is only ever called through a
// //go:noinline Go wrapper whose body is that one call, so the wrapper's
// prologue checks the stack guard the runtime poisons to request a stop
// before every tile. It type-checks the package's non-test files for each
// build (arm64, amd64, purego) and fails when
//
//   - an assembly kernel is called by name,
//   - a microkernel's f32 or q8 field is called directly, or
//   - any value of a kernel's function type is called anywhere but inside
//     such a wrapper (which also catches f := mk.f32; f(...)),
//
// and requires both wrappers to exist and to be used by forwardPacked, so the
// check cannot pass vacuously.
func TestEveryAssemblyCallIsAPreemptionPoint(t *testing.T) {
	// One importer for every build: it type-checks the standard library
	// packages once, from source, for this host.
	fset := token.NewFileSet()
	imp := importer.ForCompiler(fset, "source", nil)
	for _, b := range []struct {
		goarch string
		tags   []string
	}{
		{"arm64", nil},
		{"amd64", nil},
		{"arm64", []string{"purego"}},
	} {
		name := b.goarch
		if len(b.tags) > 0 {
			name += "+" + strings.Join(b.tags, ",")
		}
		t.Run(name, func(t *testing.T) {
			violations, wrapped := checkPreemptionStructure(t, fset, imp, b.goarch, b.tags)
			for _, v := range violations {
				t.Error(v)
			}
			hasAsm := len(b.tags) == 0
			for _, field := range kernelFieldTypes {
				if hasAsm && wrapped[field] == 0 {
					t.Errorf("forwardPacked calls no //go:noinline wrapper for the %s kernel: the check found nothing to hold", field)
				}
			}
		})
	}
}

// checkPreemptionStructure returns the rule violations for one build and,
// per kernel field, how many wrapper calls pass that field's kernel.
func checkPreemptionStructure(t *testing.T, fset *token.FileSet, imp types.Importer, goarch string, tags []string) ([]string, map[string]int) {
	t.Helper()
	ctx := build.Default
	ctx.GOOS, ctx.GOARCH, ctx.BuildTags, ctx.CgoEnabled = "linux", goarch, tags, false
	pkg, err := ctx.ImportDir(".", 0)
	if err != nil {
		t.Fatal(err)
	}
	var files []*ast.File
	for _, name := range pkg.GoFiles {
		f, err := parser.ParseFile(fset, filepath.Join(pkg.Dir, name), nil, parser.ParseComments)
		if err != nil {
			t.Fatal(err)
		}
		files = append(files, f)
	}
	info := &types.Info{
		Types:      map[ast.Expr]types.TypeAndValue{},
		Defs:       map[*ast.Ident]types.Object{},
		Uses:       map[*ast.Ident]types.Object{},
		Selections: map[*ast.SelectorExpr]*types.Selection{},
	}
	conf := types.Config{Importer: imp}
	if _, err := conf.Check("tensor", fset, files, info); err != nil {
		t.Fatalf("type-check for %s %v: %v", goarch, tags, err)
	}

	// The kernel function types, from the microkernel struct.
	var mkStruct *types.Struct
	for ident, obj := range info.Defs {
		if tn, ok := obj.(*types.TypeName); ok && ident.Name == "microkernel" {
			mkStruct, _ = tn.Type().Underlying().(*types.Struct)
		}
	}
	if mkStruct == nil {
		t.Fatal("no microkernel struct")
	}
	fieldType := map[string]types.Type{}
	for i := 0; i < mkStruct.NumFields(); i++ {
		for _, name := range kernelFieldTypes {
			if f := mkStruct.Field(i); f.Name() == name {
				fieldType[name] = f.Type()
			}
		}
	}
	kernelTypeOf := func(typ types.Type) string {
		for name, ft := range fieldType {
			if typ != nil && types.Identical(typ, ft) {
				return name
			}
		}
		return ""
	}

	// Assembly functions: declared without a body.
	asm := map[types.Object]bool{}
	for _, f := range files {
		for _, d := range f.Decls {
			if fd, ok := d.(*ast.FuncDecl); ok && fd.Body == nil {
				asm[info.Defs[fd.Name]] = true
			}
		}
	}

	var violations []string
	wrapped := map[string]int{}
	pos := func(n ast.Node) string {
		p := fset.Position(n.Pos())
		return filepath.Base(p.Filename) + ":" + strconv.Itoa(p.Line)
	}
	for _, f := range files {
		for _, d := range f.Decls {
			fd, ok := d.(*ast.FuncDecl)
			if !ok || fd.Body == nil {
				continue
			}
			wrapper := isPreemptionWrapper(fd, info, kernelTypeOf)
			ast.Inspect(fd.Body, func(n ast.Node) bool {
				call, ok := n.(*ast.CallExpr)
				if !ok {
					return true
				}
				fun := ast.Unparen(call.Fun)
				switch fn := fun.(type) {
				case *ast.Ident:
					if obj := info.Uses[fn]; asm[obj] && kernelTypeOf(obj.Type()) != "" {
						violations = append(violations, pos(call)+": "+fd.Name.Name+" calls the assembly kernel "+fn.Name+" directly, with no preemption point before it")
						return true
					}
				case *ast.SelectorExpr:
					if sel := info.Selections[fn]; sel != nil && sel.Kind() == types.FieldVal && kernelTypeOf(sel.Type()) != "" {
						violations = append(violations, pos(call)+": "+fd.Name.Name+" calls the microkernel field "+fn.Sel.Name+" directly, with no preemption point before it")
						return true
					}
				}
				if kind := kernelTypeOf(info.TypeOf(fun)); kind != "" && wrapper == "" {
					violations = append(violations, pos(call)+": "+fd.Name.Name+" calls a "+kind+" kernel value outside a //go:noinline wrapper")
				}
				// A wrapper call: count the kernels it is handed.
				if obj, ok := fun.(*ast.Ident); ok && fd.Name.Name == "forwardPacked" {
					if callee, ok := info.Uses[obj].(*types.Func); ok {
						if decl := funcDecl(files, info, callee); decl != nil {
							if kind := isPreemptionWrapper(decl, info, kernelTypeOf); kind != "" {
								wrapped[kind]++
							}
						}
					}
				}
				return true
			})
		}
	}
	return violations, wrapped
}

// isPreemptionWrapper reports the kernel field a function wraps: it must be
// marked //go:noinline and its body must be exactly one call of one of its
// own parameters, of a kernel function type.
func isPreemptionWrapper(fd *ast.FuncDecl, info *types.Info, kernelTypeOf func(types.Type) string) string {
	if fd.Doc == nil || fd.Body == nil || len(fd.Body.List) != 1 {
		return ""
	}
	noinline := false
	for _, c := range fd.Doc.List {
		noinline = noinline || strings.TrimSpace(c.Text) == "//go:noinline"
	}
	if !noinline {
		return ""
	}
	stmt, ok := fd.Body.List[0].(*ast.ExprStmt)
	if !ok {
		return ""
	}
	call, ok := stmt.X.(*ast.CallExpr)
	if !ok {
		return ""
	}
	id, ok := ast.Unparen(call.Fun).(*ast.Ident)
	if !ok {
		return ""
	}
	v, ok := info.Uses[id].(*types.Var)
	if !ok {
		return ""
	}
	for _, field := range fd.Type.Params.List {
		for _, name := range field.Names {
			if info.Defs[name] == v {
				return kernelTypeOf(v.Type())
			}
		}
	}
	return ""
}

func funcDecl(files []*ast.File, info *types.Info, fn *types.Func) *ast.FuncDecl {
	for _, f := range files {
		for _, d := range f.Decls {
			if fd, ok := d.(*ast.FuncDecl); ok && info.Defs[fd.Name] == fn {
				return fd
			}
		}
	}
	return nil
}
