library(reticulate)
py_install("pytorch")
py_install("transformers")

import("torch")
py_config()
py_run_string("from transformers import AutoTokenizer, AutoModelWithLMHead")
py_run_string("tokenizer = AutoTokenizer.from_pretrained('t5-base')")

py_run_string("""text = ('text')""")

sequence = ("""In May, Churchill was still generally unpopular with many Conservatives and probably most of the Labour Party. Chamberlain "
              "remained Conservative Party leader until October when ill health forced his resignation. By that time, Churchill had won the "
              "doubters over and his succession as party leader was a formality."
              " "
              "He began his premiership by forming a five-man war cabinet which included Chamberlain as Lord President of the Council, "
              "Labour leader Clement Attlee as Lord Privy Seal (later as Deputy Prime Minister), Halifax as Foreign Secretary and Labours "
              "Arthur Greenwood as a minister without portfolio. In practice, these five were augmented by the service chiefs and ministers "
              "who attended the majority of meetings. The cabinet changed in size and membership as the war progressed, one of the key "
              "appointments being the leading trades unionist Ernest Bevin as Minister of Labour and National Service. In response to "
              "previous criticisms that there had been no clear single minister in charge of the prosecution of the war, Churchill created "
              "and took the additional position of Minister of Defence, making him the most powerful wartime Prime Minister in British "
              "history. He drafted outside experts into government to fulfil vital functions, especially on the Home Front. These included "
              "personal friends like Lord Beaverbrook and Frederick Lindemann, who became the governments scientific advisor."""")

py_run_string("inputs = tokenizer.encode('summarize: ' + text, return_tensors = 'pt', max_length = 512, truncation = True)")

