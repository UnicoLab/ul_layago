import unittest
from generate_synthetic import generate


class SyntheticTests(unittest.TestCase):
    def test_story_and_domain_isolation_and_counterfactuals(self):
        rows = generate(12)
        self.assertEqual(rows, generate(12))
        self.assertEqual(len({row['id'] for row in rows}), len(rows))
        stories = {}
        for row in rows:
            self.assertEqual(stories.setdefault(row['story_id'], row['split']), row['split'])
            if row['label'] == 0:
                self.assertEqual(row['input']['question'], row['input']['evidence'][0]['text'])
            else:
                self.assertNotEqual(row['input']['question'], row['input']['evidence'][0]['text'])
        fit_domains = {r['domain_id'] for r in rows if r['split'] in ('train', 'dev')}
        self.assertFalse(fit_domains & {r['domain_id'] for r in rows if r['split'] == 'heldout_domain'})
        self.assertEqual({r['label'] for r in rows}, {0, 1, 2})

    def test_small_split_refused(self):
        with self.assertRaises(ValueError):
            generate(1)
