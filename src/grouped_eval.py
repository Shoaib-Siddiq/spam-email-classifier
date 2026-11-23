"""Evaluate the repository's SMS classifier with repeated normalized text grouped."""
import json
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from src.data import clean_text, load_data
from src.models import build_vectorizer, RANDOM_STATE


def main():
    frame = load_data()
    text = frame.message.map(clean_text)
    model = make_pipeline(build_vectorizer(), SGDClassifier(loss='modified_huber',
                          random_state=RANDOM_STATE, max_iter=2000))
    results = {'messages': len(text), 'distinct_normalized_messages': text.nunique()}
    for label, splits in [('ordinary', StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE).split(text, frame.label)),
                          ('grouped', StratifiedGroupKFold(5, shuffle=True, random_state=RANDOM_STATE).split(text, frame.label, text))]:
        pred = cross_val_predict(model, text, frame.label, cv=list(splits))
        p, r, f, _ = precision_recall_fscore_support(frame.label, pred, average='binary')
        results[label] = dict(spam_precision=p, spam_recall=r, spam_f1=f)
    Path('reports/grouped_eval.json').write_text(json.dumps(results, indent=2)+'\n')
    print(results)


if __name__ == '__main__':
    main()
