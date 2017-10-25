import language_check
tool = language_check.LanguageTool('en-US')
text = "And specifically doing it in another language this can be tricky at 5 yeah, so if you guys have been listening to the podcast for a while. You're going to know that we have done quite a bit of traveling and and we're we've been we've been to Hawaii we've been to California. We've been to Costa Rica yeah, Costa Rica. You can drive La panama ecuador and coming. We have a trip plan to ecuador again and to Columbia."
matches = tool.check(text)
len(matches)
correct = language_check.correct(text, matches)
print(correct)