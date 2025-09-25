cat sentences.txt | sed -f en/tokenizer.sed > sentences.tok
candc-1.00/bin/candc --models candc-1.00/models --candc-printer xml --input sentences.tok > sentences.candc.xml
python en/candc2transccg.py sentences.candc.xml > sentences.xml
python scripts/semparse.py sentences.xml en/semantic_templates_en_emnlp2015.yaml sentences.sem.xml > semparse.log 2>&1