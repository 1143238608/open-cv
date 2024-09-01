from transformers import pipeline

classifier = pipeline(task='sentiment-analysis')
res = classifier("Today is a nice day!")
print(res)
