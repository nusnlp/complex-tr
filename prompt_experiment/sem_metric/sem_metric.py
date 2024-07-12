# Copyright 2020 The HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Custom metric. """

import datasets

from .evaluate import (
    apply_no_ans_threshold,
    find_all_best_thresh,
    get_raw_scores,
    make_eval_dict,
    make_qid_to_has_ans,
    merge_eval,
)


_CITATION = """
"""

_DESCRIPTION = """
"""

_KWARGS_DESCRIPTION = """
Computes Strict EM scores (token F1, EM, answer F1 and Strict EM).
Args:
    predictions: List of triple for question-answers to score with the following elements:
        - the question-answer 'id' field as given in the references (see below)
        - the text of the answer
        - the probability that the question has no answer
    references: List of question-answers dictionaries with the following key-values:
            - 'id': id of the question-answer pair (see above),
            - 'answers': a list of Dict {'text': text of the answer as a string}
    no_answer_threshold: float
        Probability threshold to decide that a question has no answer.
Returns:
    'exact': Exact match (the normalized answer exactly match the gold answer)
    'f1': The F-score of predicted tokens versus the gold answer
    'total': Number of score considered
    'HasAns_exact': Exact match (the normalized answer exactly match the gold answer)
    'HasAns_f1': The F-score of predicted tokens versus the gold answer
    'HasAns_total': Number of score considered
    'NoAns_exact': Exact match (the normalized answer exactly match the gold answer)
    'NoAns_f1': The F-score of predicted tokens versus the gold answer
    'NoAns_total': Number of score considered
    'best_exact': Best exact match (with varying threshold)
    'best_exact_thresh': No-answer probability threshold associated to the best exact match
    'best_f1': Best F1 (with varying threshold)
    'best_f1_thresh': No-answer probability threshold associated to the best F1
Examples:

    >>> predictions = [{'prediction_text': '1976', 'id': '56e10a3be3433e1400422b22', 'no_answer_probability': 0.}]
    >>> references = [{'answers': {'answer_start': [97], 'text': ['1976']}, 'id': '56e10a3be3433e1400422b22'}]
    >>> squad_v2_metric = datasets.load_metric("squad_v2")
    >>> results = squad_v2_metric.compute(predictions=predictions, references=references)
    >>> print(results)
    {'exact': 100.0, 'f1': 100.0, 'total': 1, 'HasAns_exact': 100.0, 'HasAns_f1': 100.0, 'HasAns_total': 1, 'best_exact': 100.0, 'best_exact_thresh': 0.0, 'best_f1': 100.0, 'best_f1_thresh': 0.0}
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Squad_SEM(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": {
                        "id": datasets.Value("string"),
                        "prediction_text": datasets.Value("string"),
                        "no_answer_probability": datasets.Value("float32"),
                    },
                    "references": {
                        "id": datasets.Value("string"),
                        "answers": datasets.features.Sequence(
                            {"text": datasets.Value("string")}
                        ),
                    },
                }
            ),
            codebase_urls=["https://rajpurkar.github.io/SQuAD-explorer/"],
            reference_urls=["https://rajpurkar.github.io/SQuAD-explorer/"],
        )

    def _compute(self, predictions, references, no_answer_threshold=1.0):
        no_answer_probabilities = {p["id"]: p["no_answer_probability"] for p in predictions}
        dataset = [{"paragraphs": [{"qas": references}]}]
        predictions = {p["id"]: p["prediction_text"] for p in predictions}

        qid_to_has_ans = make_qid_to_has_ans(dataset)  # maps qid to True/False
        has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
        no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]

        exact_raw, f1_raw, strict_exact_raw, answer_f1_raw = get_raw_scores(dataset, predictions)
        
        exact_thresh = apply_no_ans_threshold(exact_raw, no_answer_probabilities, qid_to_has_ans, no_answer_threshold)
        f1_thresh = apply_no_ans_threshold(f1_raw, no_answer_probabilities, qid_to_has_ans, no_answer_threshold)
        strict_exact_thresh = apply_no_ans_threshold(strict_exact_raw, no_answer_probabilities, qid_to_has_ans, no_answer_threshold)
        answer_f1_thresh = apply_no_ans_threshold(answer_f1_raw, no_answer_probabilities, qid_to_has_ans, no_answer_threshold)
        out_eval = make_eval_dict(exact_thresh, f1_thresh, strict_exact_thresh, answer_f1_thresh)

        if has_ans_qids:
            has_ans_eval = make_eval_dict(exact_thresh, f1_thresh,strict_exact_thresh, answer_f1_thresh, qid_list=has_ans_qids)
            merge_eval(out_eval, has_ans_eval, "HasAns")
        if no_ans_qids:
            no_ans_eval = make_eval_dict(exact_thresh, f1_thresh, strict_exact_thresh, answer_f1_thresh, qid_list=no_ans_qids)
            merge_eval(out_eval, no_ans_eval, "NoAns")
        find_all_best_thresh(out_eval, predictions, exact_raw, f1_raw, no_answer_probabilities, qid_to_has_ans)
        return dict(out_eval)
