import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import audit_training_data as a


def row(ident='a',split='train',text='Original warehouse holds five boxes.',label=0,**extra):
    raw={'id':ident,'split':split,'lang':'en','instance':'R','task':'r_support','task_type':'choice',
         'input':{'question':'Does the claim follow?','evidence':[{'id':'s','text':text}]},
         'options':[{'key':'yes','text':'Supported.'},{'key':'no','text':'Not supported.'}],
         'label':label,'label_source':'template','story_id':'story-'+ident,'domain_id':'warehouse','generator_version':'fixture-v1','seed':1}
    raw.update(extra);return raw


class AuditTests(unittest.TestCase):
    def test_exact_copies_and_story_group_isolation(self):
        one=row(meta={'group_id':'translation-family'})
        two=row('b','dev',story_id=one['story_id'],meta={'group_id':'translation-family'})
        r=a.audit_rows([one,two]);counts=r['findings']['counts']
        self.assertEqual(counts['cross_split_story_id'],1)
        self.assertEqual(counts['cross_split_group_id'],1)
        self.assertEqual(counts['exact_source_copy'],1)
        self.assertEqual(counts['exact_rendered_state_copy'],1)
        self.assertNotIn(one['input']['evidence'][0]['text'],json.dumps(r))

    def test_duplicate_id_same_split_is_reported(self):
        r=a.audit_rows([row(),row(text='Different.')])
        self.assertEqual(r['findings']['counts']['duplicate_id'],1)
        self.assertEqual(r['status'],'blocked')

    def test_source_leak_survives_changed_task_question_and_label(self):
        one=row();two=row('b','dev',task='r_other',label=1)
        two['input']['question']='Entirely different instructions.'
        self.assertEqual(a.audit_rows([one,two])['findings']['counts']['exact_source_copy'],1)

    def test_conflicting_labels_only_same_complete_task_input_and_catalog(self):
        one=row();two=row('b',label=1)
        self.assertEqual(a.audit_rows([one,two])['findings']['counts']['conflicting_hard_labels'],1)
        two['input']['question']='Counterfactual different claim'
        self.assertNotIn('conflicting_hard_labels',a.audit_rows([one,two])['findings']['counts'])
        two['input']['question']=one['input']['question'];two['options'][0]['text']='Different option meaning'
        self.assertNotIn('conflicting_hard_labels',a.audit_rows([one,two])['findings']['counts'])

    def test_fixed_catalog_and_question_are_not_source_copy(self):
        r=a.audit_rows([row(),row('b','dev',text='A different warehouse with ten items.')])
        self.assertNotIn('exact_source_copy',r['findings']['counts'])
        self.assertNotIn('near_source_copy',r['findings']['counts'])
        q=row(instance='Q',task='q_metric',input={'question':'Fixed instructions','catalog':[{'key':'sales','label':'Sales'}]})
        qr=a.audit_rows([q,row('c','dev',instance='Q',task='q_metric',input=q['input'])])
        self.assertNotIn('exact_source_copy',qr['findings']['counts'])
        self.assertEqual(qr['findings']['counts']['exact_rendered_state_copy'],1)

    def test_dev_halves_exact_source_leak_separate_from_raw_split(self):
        names={}
        for i in range(30):names.setdefault(a.dev_half(str(i)),str(i))
        rows=[row('a','dev',story_id=names['dev_cal']),row('b','dev',story_id=names['dev_conf'])]
        self.assertEqual(a.audit_rows(rows)['findings']['counts']['exact_source_copy'],1)

    def test_locked_rows_never_inspected_or_rendered(self):
        class Sealed(dict):
            def __getitem__(self,k):raise AssertionError('locked content accessed')
            def get(self,k,*args):
                if k=='split':return 'locked_test'
                raise AssertionError('locked content accessed')
        r=a.audit_rows([row(),Sealed()])
        self.assertEqual(r['locked_test']['skipped_rows'],1)
        self.assertEqual(r['valid_rows'],1)
        self.assertEqual(r['surface_probe']['status'],'not_requested')

    def test_loader_discards_locked_payload_after_envelope_decode(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data.jsonl';p.write_text(json.dumps(row('locked','locked_test',text='SEALED TEXT',label=1))+'\n')
            loaded,files=a.load_rows([p]);self.assertEqual(loaded,[{'split':'locked_test'}]);self.assertEqual(files[0]['rows'],1)
            full,_=a.load_rows([p],unseal_locked=True);self.assertIn('input',full[0])

    def test_unseal_requires_explicit_durable_log(self):
        with self.assertRaises(ValueError):a.audit_rows([row('l','locked_test')],unseal_locked=True)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data.jsonl';p.write_text(json.dumps(row('l','locked_test'))+'\n')
            out=Path(d)/'report.json';md=Path(d)/'report.md';log=Path(d)/'access.jsonl'
            self.assertEqual(a.main([str(p),'--json-out',str(out),'--markdown-out',str(md),'--unseal-locked','--access-log',str(log)]),0)
            self.assertEqual(json.loads(log.read_text())['event'],'training_data_audit_locked_access')
            self.assertEqual(json.loads(out.read_text())['locked_test']['access'],'explicit_unsealed')

    def test_calibration_matches_go_score_family_defaults_and_story_counts(self):
        self.assertEqual(a.calibration_target(),176)
        self.assertEqual(a.calibration_target(delta=.2),135)
        self.assertEqual(a.calibration_target(start_errors=1),93)
        name=next(str(i) for i in range(20) if a.dev_half(str(i))=='dev_conf')
        rows=[row(str(i),'dev',story_id=name,text=str(i)) for i in range(10)]
        ready=a.audit_rows(rows)['calibration_readiness']
        self.assertEqual(ready['tasks'][0]['dev_conf'],10)
        self.assertEqual(ready['tasks'][0]['dev_conf_unique_stories'],1)
        self.assertIn('correlated',ready['row_correlation_warning'])

    def test_calibration_pathologies_bounded(self):
        for options in ({'alpha':0},{'delta':float('nan')},{'n_min':True},{'start_errors':101},{'alpha':5e-324},{'delta':5e-324}):
            with self.subTest(options=options),self.assertRaises(ValueError):a.calibration_target(**options)

    def test_near_source_copy_including_long_contract_tail_change(self):
        text=' '.join('clause'+str(i) for i in range(100))
        rows=[row(text=text),row('b','dev',text=text+' different ending')]
        r=a.audit_rows(rows)
        self.assertEqual(r['findings']['counts']['near_source_copy'],1)
        self.assertEqual(r['near_copy']['coverage'],'heuristic_bounded_not_exhaustive')
        self.assertEqual(r,a.audit_rows(rows))

    def test_near_copy_budgets_disclosed_not_quadratic(self):
        text=' '.join('word'+str(i) for i in range(60))
        rows=[row(str(i),'train' if i%2 else 'dev',text=text+' unique'+str(i)) for i in range(15)]
        config=a.NearConfig(max_comparisons=2,max_parts=6,max_posting=2,max_shingles=20)
        r=a.audit_rows(rows,near_config=config)['near_copy']
        self.assertLessEqual(r['comparisons'],2)
        self.assertGreater(r['sampled_parts'],0)
        self.assertGreater(r['skipped']['part_budget'],0)

    def test_unknown_labels_options_and_bool_indices_refused(self):
        for value in (True,-1,2,'yes'):
            with self.subTest(value=value):self.assertEqual(a.audit_rows([row(label=value)])['invalid_rows'],1)
        catalog={'r_support':['yes','no']}
        self.assertEqual(a.audit_rows([row()],catalog=catalog)['valid_rows'],1)
        bad=row();bad['options'][0]['key']='invented'
        self.assertEqual(a.audit_rows([bad],catalog=catalog)['invalid_rows'],1)

    def test_soft_label_contract_and_yes_no(self):
        self.assertEqual(a.audit_rows([row(label_dist=[.6,.4])])['valid_rows'],1)
        self.assertEqual(a.audit_rows([row(split='dev',label_dist=[.6,.4])])['invalid_rows'],1)
        self.assertEqual(a.audit_rows([row(label_dist=[.4,.6])])['invalid_rows'],1)
        yes=row(task_type='yes_no',options=[],label=True)
        self.assertEqual(a.audit_rows([yes])['counts']['task_labels']['r_support'],{'true':1})

    def test_leftovers_and_metadata_round_shortcut(self):
        rows=[row(str(i),text='{{INSERT_HERE}}',label=i%2,seed=i%2) for i in range(8)]
        r=a.audit_rows(rows)
        self.assertEqual(r['findings']['counts']['template_leftover'],8)
        self.assertIn('seed',[m['field'] for m in r['metadata_shortcuts']])

    def test_surface_probe_train_fitted_only_explicit_dev(self):
        fit=[row(str(i),text=('x '*4 if i%2==0 else 'long '*200),label=i%2) for i in range(20)]
        dev=[row('dev'+str(i),'dev',text=('z '*4 if i%2==0 else 'test '*200),label=i%2) for i in range(10)]
        held=[row('test'+str(i),'heldout_domain',text='ignored',label=1) for i in range(10)]
        before=copy.deepcopy(fit+dev+held)
        result=a.audit_rows(fit+dev+held,surface_eval='dev')['surface_probe']
        task=result['tasks'][0]
        self.assertEqual(task['evaluation'],10);self.assertEqual(task['correct'],10)
        self.assertGreater(task['accuracy'],task['train_majority_accuracy'])
        self.assertTrue(task['shortcut_warning'])
        self.assertNotEqual(a.audit_rows(fit+dev,surface_eval='dev')['status'],'checks_passed_with_limits')
        changed=copy.deepcopy(dev)
        for r in changed:r['label']=1-r['label'];r['input']['evidence'][0]['text']='new very different evaluation content'
        again=a.audit_rows(fit+changed,surface_eval='dev')['surface_probe']['tasks'][0]
        self.assertEqual(task['train_feature_means'],again['train_feature_means'])
        self.assertEqual(fit+dev+held,before)
        self.assertIn('Surface accuracy',a.markdown(a.audit_rows(fit+dev,surface_eval='dev')))
        with self.assertRaises(ValueError):a.audit_rows(fit,surface_eval='locked_test')

    def test_report_bounded_findings_and_does_not_replace_input_or_report(self):
        r=a.audit_rows([row(str(i),text='TODO') for i in range(300)])
        self.assertLessEqual(len(r['findings']['items']),a.MAX_FINDINGS);self.assertGreater(r['findings']['omitted'],0)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input.jsonl';p.write_text(json.dumps(row())+'\n');original=p.read_bytes()
            out=Path(d)/'out.json';md=Path(d)/'out.md'
            with self.assertRaises(SystemExit):a.main([str(p),'--json-out',str(p),'--markdown-out',str(md)])
            self.assertEqual(a.main([str(p),'--json-out',str(out),'--markdown-out',str(md)]),0)
            with self.assertRaises(SystemExit):a.main([str(p),'--json-out',str(out),'--markdown-out',str(md)])
            self.assertEqual(p.read_bytes(),original)

    def test_loader_strict_json_and_resource_bound(self):
        for body in ('{"split":"train","split":"locked_test"}', '{"x":NaN}'):
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'data';p.write_text(body+'\n')
                with self.assertRaises(ValueError):a.load_rows([p])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data';p.write_text(' ' * 30+'\n')
            with patch.object(a,'MAX_LINE',10),self.assertRaises(ValueError):a.load_rows([p])

    def test_arbitrary_meta_and_unicode_render(self):
        r=a.audit_rows([row(meta=['ignored'])]);self.assertEqual(r['valid_rows'],1)
        raw=row(input={'question':'\u00a0Q\n','evidence':[{'id':'\tα','text':' Épreuve 😀\n'}]})
        self.assertEqual(a.render_state(raw),'question: Q\nevidence:\n- α: Épreuve 😀')
        q=row(instance='Q',task='q_metric',input={'question':' Q ','catalog':[{'key':'k','labels':{'fr':'Énergie','en':'Energy'},'unit':' kW ','dimensions':[' x '],'scope_values':{'x':[' a ','b']}}]})
        self.assertEqual(a.render_state(q),'question: Q\ncatalog:\n- k: Energy | Énergie; unit: kW; dimensions: x; scope: x=a/b')
        invalid=a.audit_rows([row(meta={'group_id':'\ud800'})])
        self.assertEqual(invalid['findings']['counts']['invalid_group_identifier'],1)

    def test_invalid_empty_object_options_not_coerced_and_scalar_evidence(self):
        self.assertEqual(a.audit_rows([row(task_type='yes_no',options={},label=True)])['invalid_rows'],1)
        raw=row(input={'question':'Q','evidence':[123,None,False]})
        self.assertEqual(a.parse_row(raw)['parts'],['123','None','False'])

    def test_model_generated_text_requires_provenance(self):
        self.assertEqual(a.audit_rows([row(label_source='llm_translation')])['invalid_rows'],1)
        self.assertEqual(a.audit_rows([row(label_source='llm_translation',provenance={'llm_interaction_ids':['a'*64]})])['valid_rows'],1)

    def test_heldout_domains_and_languages_cannot_overlap_fit(self):
        data=[row(),row('d','heldout_domain',text='different'),row('l','heldout_language',text='third')]
        r=a.audit_rows(data)
        self.assertEqual(r['findings']['counts']['heldout_domain_overlap'],1)
        self.assertEqual(r['findings']['counts']['heldout_language_overlap'],1)
        self.assertEqual(r['status'],'blocked')


if __name__=='__main__':unittest.main()
