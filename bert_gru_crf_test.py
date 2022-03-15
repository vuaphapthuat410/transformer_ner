from transformers import XLMRobertaTokenizerFast
from models import BertGRUCRF

model = BertGRUCRF.from_pretrained('../drive/MyDrive/NLP/MiniLM/MiniLM-GRU-CRF', num_labels=9)
tokenizer = XLMRobertaTokenizerFast.from_pretrained("microsoft/Multilingual-MiniLM-L12-H384", do_lower_case=False)
id2label = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]


def decode(label_ids, input_ids, offsets_mapping, id2label):
    result = [[]]
    for k in range(len(label_ids)):
        words = []
        labels = []
        for i in range(len(label_ids[k])):
            word = tokenizer.convert_ids_to_tokens([int(input_ids[k][i])], skip_special_tokens=True)
            if word == []:
            	continue
            else:
            	word = word[0]

            if not word.startswith('▁'):
                words[-1] += word 
                labels[-1] = id2label[int(label_ids[k][i])] if labels[-1] == "O" else labels[-1]
            else:
                words.append(word)
                labels.append(id2label[int(label_ids[k][i])])

        for word, tag in zip(words, labels):
            result[k].append({'word': word, 'tag': tag})
                
    return result


corpus = [
    ' "Ông Zelensky đã rời Ukraine. Các nghị sĩ Verkhovna Rada (Quốc hội Ukraine) nói rằng họ không thể gặp ông ở thành phố Lviv. Ông ấy hiện đang ở Ba Lan", Hãng tin Sputnik dẫn lời Chủ tịch Duma Quốc gia Nga Vyacheslav Volodin khẳng định. Hãng tin này của Nga tiếp tục đưa thông tin từ chính trị gia đối lập Ilya Kiva của Ukraine thậm chí còn đoan chắc ông Zelensky đang trú trong Đại sứ quán Mỹ ở Ba Lan. Ngược lại, Đài Russia Today đưa tin Quốc hội Ukraine đã bác bỏ thông tin này. "Quốc hội Ukraine tuyên bố Tổng thống Zelensky vẫn ở Kiev sau khi có nhiều thông tin rằng ông đã sang Ba Lan", đài này cho biết. Phía Tổng thống Zelensky chưa lên tiếng về thông tin trên. Bản thân ông cũng từng nhận định ông và gia đình là "mục tiêu số 1" vì Nga muốn tiêu diệt Ukraine về mặt chính trị bằng cách tiêu diệt nguyên thủ quốc gia của nước này. Trước đó, một số nước phương Tây đã lên tiếng lo ngại về an toàn của ông Zelensky. Ngoại trưởng Pháp Jean-Yves Le Drian nói Paris sẵn sàng giúp nhà lãnh đạo Ukraine khi cần thiết. Mỹ cũng từng đề nghị đưa ông Zelensky rời Kiev, tuy nhiên ông từ chối. "Cuộc chiến đang diễn ra ở đây. Điều tôi cần là đạn dược chứ không phải chạy đi. Người dân Ukraine tự hào về tổng thống của họ", Đài CNN của Mỹ dẫn tiết lộ từ Đại sứ quán Ukraine ở Anh. Ông Zelensky cũng xuất hiện trong một video sau đó khẳng định "đừng tin vào tin giả" vì ông vẫn ở Kiev. "Tôi vẫn ở đây. Chúng tôi sẽ không buông vũ khí. Chúng tôi sẽ bảo vệ đất nước vì vũ khí của chúng tôi là sự thật, sự thật rằng đây là mảnh đất, đất nước, con cái của chúng tôi và chúng tôi sẽ bảo vệ tất cả điều đó", ông nói.'
]

inputs = tokenizer(corpus, max_length=512, padding=True, truncation=True, return_tensors='pt',
                   return_offsets_mapping=True)
offset_mapping = inputs.pop("offset_mapping").cpu().numpy().tolist()

outputs = model(**inputs)
# print(decode(outputs[1].numpy().tolist(), inputs['input_ids'].numpy().tolist(), offset_mapping, id2label))
decoded_output = decode(outputs[1].numpy().tolist(), inputs['input_ids'].numpy().tolist(), offset_mapping, id2label)
for i in decoded_output:
  for j in i:
    print(j)