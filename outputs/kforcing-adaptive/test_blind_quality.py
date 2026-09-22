"""Synthetic-only checks of frozen scoring rules. No human responses are generated."""
import copy
import json
from pathlib import Path
import unittest
from analyze_blind_quality import analyze, validate_packet, read

ROOT = Path(__file__).resolve().parent / 'results/blind-quality'

class BlindQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packet=read(ROOT/'reviewer/tasks.json'); cls.key=read(ROOT/'private/key.json')
        cls.main=[i for i,k in cls.key['items'].items() if k['kind']=='main']

    def fixture(self, alias, preferred='split22'):
        answers=[]
        for task in self.packet['tasks']:
            k=self.key['items'][task['id']]
            if k['kind']=='identical' or preferred=='TIE': choice='TIE'
            elif preferred=='SKIP': choice='SKIP'
            else: choice='A' if k['A_method']==preferred else 'B'
            answers.append(dict(task_id=task['id'],choice=choice,both_bad=False,comment='Synthetic test only'))
        return dict(schema_version=1,packet_id=self.packet['packet_id'],tasks_sha256=self.packet['tasks_sha256'],
                    data_origin='synthetic',reviewer=dict(alias=alias,human_confirmed=False),complete=True,answers=answers)

    def pair(self, preferred='split22'):
        return [self.fixture('synthetic-one',preferred),self.fixture('synthetic-two',preferred)]

    def run_synthetic(self, ratings):
        result=analyze(self.packet,self.key,ratings,allow_synthetic=True)
        self.assertEqual(result['human_ratings_received'],0)
        self.assertEqual(result['quality_result'],'synthetic_test_only')
        return result

    def test_extreme_preferences_and_ties(self):
        for preferred,value in [('split22',1),('keep4',-1),('TIE',0)]:
            r=self.run_synthetic(self.pair(preferred))
            self.assertEqual(r['main']['mean_net_split_preference'],value)
            self.assertEqual(r['main']['source_bootstrap_95ci'],[value,value])
            self.assertEqual(r['main']['all_source_missing_bounds'],[value,value])
            self.assertEqual(r['main']['complete_case_votes']['n'],128)
            self.assertEqual(r['pairwise_raw_agreement'][0]['agreement'],1)

    def test_direction_adjusted_swap_and_identical_controls(self):
        r=self.run_synthetic(self.pair())
        for qc in r['reviewer_checks']:
            self.assertEqual(qc['swap_consistency'],1)
            self.assertEqual(qc['identical_tie_rate'],1)
            self.assertEqual(qc['swap_comparable'],8)

    def test_opposite_reviewers_average_within_source(self):
        r=self.run_synthetic([self.fixture('synthetic-one'),self.fixture('synthetic-two','keep4')])
        self.assertEqual(r['main']['mean_net_split_preference'],0)
        self.assertEqual(r['main']['source_bootstrap_95ci'],[0,0])
        self.assertEqual(r['main']['complete_case_votes']['split_rate'],.5)
        self.assertEqual(r['pairwise_raw_agreement'][0]['agreement'],0)

    def test_complete_case_threshold_and_missing_bounds(self):
        for skipped,sufficient in [(16,True),(17,False)]:
            ratings=self.pair(); excluded=set(self.main[:skipped])
            for answer in ratings[0]['answers']:
                if answer['task_id'] in excluded: answer['choice']='SKIP'
            r=self.run_synthetic(ratings)['main']
            self.assertEqual(r['complete_sources'],64-skipped)
            self.assertEqual(r['missing_votes'],skipped)
            self.assertEqual(r['mean_net_split_preference'],1 if sufficient else None)
            self.assertEqual(r['all_source_missing_bounds'],[(128-2*skipped)/128,1])
            self.assertEqual(r['complete_case_votes']['n'],(64-skipped)*2)

    def test_all_missing_is_not_tie(self):
        r=self.run_synthetic(self.pair('SKIP'))['main']
        self.assertIsNone(r['mean_net_split_preference'])
        self.assertIsNone(r['source_bootstrap_95ci'])
        self.assertEqual(r['all_source_missing_bounds'],[-1,1])
        self.assertEqual(r['all_observed_votes']['ties'],0)

    def test_nonconstant_bootstrap(self):
        ratings=self.pair(); inverse=set(self.main[:32])
        for rating in ratings:
            for a in rating['answers']:
                if a['task_id'] in inverse: a['choice']='A' if a['choice']=='B' else 'B'
        r=self.run_synthetic(ratings)['main']
        self.assertEqual(r['mean_net_split_preference'],0)
        self.assertLess(r['source_bootstrap_95ci'][0],0)
        self.assertGreater(r['source_bootstrap_95ci'][1],0)
        self.assertEqual(r,self.run_synthetic(ratings)['main'])

    def test_bad_controls_do_not_exclude_reviewer(self):
        ratings=self.pair()
        for a in ratings[0]['answers']:
            kind=self.key['items'][a['task_id']]['kind']
            if kind=='swap': a['choice']='A' if a['choice']=='B' else 'B'
            if kind=='identical': a['choice']='A'
        r=self.run_synthetic(ratings)
        self.assertEqual(r['reviewer_checks'][0]['swap_consistency'],0)
        self.assertEqual(r['reviewer_checks'][0]['identical_tie_rate'],0)
        self.assertFalse(r['reviewer_checks'][0]['excluded'])
        self.assertEqual(r['main']['mean_net_split_preference'],1)

    def test_synthetic_rejected_by_default(self):
        with self.assertRaisesRegex(ValueError,'origin rejected'): analyze(self.packet,self.key,self.pair())

    def test_duplicate_alias_unicode_normalization(self):
        ratings=self.pair(); ratings[1]['reviewer']['alias']=' ＳＹＮＴＨＥＴＩＣ－ＯＮＥ '
        with self.assertRaisesRegex(ValueError,'Duplicate reviewer'): self.run_synthetic(ratings)

    def test_wrong_packet_or_digest_rejected(self):
        for field in ['packet_id','tasks_sha256']:
            ratings=self.pair(); ratings[0][field]='wrong'
            with self.assertRaisesRegex(ValueError,'mismatch'): self.run_synthetic(ratings)

    def test_missing_duplicate_unknown_task_rejected(self):
        for change in ['missing','duplicate','unknown']:
            ratings=self.pair()
            if change=='missing': ratings[0]['answers'].pop()
            elif change=='duplicate': ratings[0]['answers'][0]=copy.deepcopy(ratings[0]['answers'][1])
            else: ratings[0]['answers'][0]['task_id']='unknown'
            with self.assertRaises(ValueError): self.run_synthetic(ratings)

    def test_partial_invalid_fields_rejected(self):
        for field,value in [('choice',None),('choice','SPLIT'),('both_bad',1),('comment',False)]:
            ratings=self.pair(); ratings[0]['answers'][0][field]=value
            with self.assertRaises(ValueError): self.run_synthetic(ratings)
        ratings=self.pair(); ratings[0]['complete']=False
        with self.assertRaisesRegex(ValueError,'Partial'): self.run_synthetic(ratings)

    def test_human_declaration_and_two_reviewers_required(self):
        ratings=self.pair()
        for r in ratings: r['data_origin']='human_self_reported'
        with self.assertRaisesRegex(ValueError,'Human declaration'): analyze(self.packet,self.key,ratings)
        with self.assertRaisesRegex(ValueError,'At least two'): self.run_synthetic(self.pair()[:1])

    def test_tampered_packet_or_key_rejected(self):
        packet=copy.deepcopy(self.packet); packet['tasks'][0]['A']+=' tampered'
        with self.assertRaisesRegex(ValueError,'digest'): validate_packet(packet,self.key)
        key=copy.deepcopy(self.key); key['items'][self.main[0]]['A_method']='unknown'
        with self.assertRaises(ValueError): validate_packet(self.packet,key)

if __name__=='__main__': unittest.main(verbosity=2)
