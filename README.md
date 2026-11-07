# ScreenAgent Prototype (DIM 716)

Prototype ของงานวิจัยตัวอย่างในเอกสาร "ตัวอย่างการออกแบบงานวิจัย Agentic AI (Module 1–5)": ระบบ multi-agent คัดกรอง abstract สำหรับ systematic review ด้านการศึกษา ประกอบด้วย Supervisor, Screener (ReAct), Critic, memory 4 ชั้น, input guard, permission gate และ audit log แบบ hash chain

> **หมายเหตุ:** ผลใน `RESULTS.md` และโฟลเดอร์ `results/` มาจาก `MockLLM` ซึ่งเป็น model จำลอง ใช้ตรวจว่า pipeline และการวิเคราะห์ทำงานถูกต้อง และใช้เพื่อการสอนเท่านั้น ไม่ใช่ผลวิจัย ห้ามนำไปอ้างอิง ต้องรันซ้ำกับ local model จริงก่อน (ดูหัวข้อ "เปลี่ยนเป็น local model จริง")

เขียนด้วย Python ล้วน (standard library) จึงรันได้ทันทีบนเครื่องใดก็ได้หรือบน Google Colab โดยไม่ต้องติดตั้งอะไร

## โครงสร้างไฟล์

| ไฟล์ | เนื้อหา | ตรงกับหัวข้อในเอกสาร |
| --- | --- | --- |
| `screenagent/data.py` | สร้างชุด abstract สังเคราะห์ 400 เรื่อง (dev 100 / test 300) พร้อม ground truth และ injection 30 เรื่อง | 2, 6.6 |
| `screenagent/memory.py` | Working (blackboard), Semantic (TF-IDF แทน ChromaDB), Episodic (มนุษย์เขียนได้คนเดียว), Procedural (freeze + hash) | 4 |
| `screenagent/safety.py` | input guard, permission gate ตาม risk tier แบบ task-bound, audit log แบบ hash chain | 6.1–6.2 |
| `screenagent/agents.py` | prompts, ToolBox, Screener, Critic, Supervisor, baseline B0–B2 | 3–5 |
| `screenagent/llm.py` | `MockLLM` (จำลอง) และ `OllamaLLM` (local model จริง) | 2 |
| `screenagent/stats.py` | Wilson CI, McNemar exact, Holm, Cohen's h, sample size แบบ paired | 6.6–6.7 |
| `run_experiment.py` | รันทุกเงื่อนไข 3 seeds, majority vote, วิเคราะห์, เขียนผลลง `results/` | 6.4–6.8 |
| `app.py` | หน้า UI แบบ Streamlit | 5 |
| `tests/test_agents.py` | unit test 23 ข้อ แยกตาม component | – |
| `RESULTS.md` | สรุปผลการทดสอบ (ภาษาไทย) | – |

## วิธีรัน

```bash
python -m unittest -v tests.test_agents      # unit tests (~0.03 วินาที)
python run_experiment.py                     # การทดลองแบบ mock (~2 วินาที)
python run_experiment.py --limit 20          # ทดลองเร็ว ๆ 20 abstracts
```

### เปิดหน้า UI (Streamlit)

```bash
pip3 install streamlit
python3 run_experiment.py        # สร้างไฟล์ผลให้แท็บ "ผลการทดลอง" (ทำครั้งเดียว)
streamlit run app.py             # เบราว์เซอร์จะเปิด http://localhost:8501 ให้อัตโนมัติ
```

หน้า UI มี 5 แท็บ: ภาพรวม, คัดกรองทีละเรื่อง (ดูการทำงานของแต่ละ agent), คิวตรวจโดยมนุษย์ (ตัดสินกรณี uncertain และอนุมัติ export), Safety (ทดสอบ injection เทียบมี/ไม่มี gate และตรวจ audit log) และผลการทดลอง เปิด/ปิด Critic, Memory, Gate, Guard ได้จากแถบซ้าย

### เปลี่ยนเป็น local model จริง (Ollama)

```bash
ollama serve &
ollama pull qwen2.5:7b-instruct
python run_experiment.py --backend ollama --model qwen2.5:7b-instruct --limit 20   # pilot ก่อน
python run_experiment.py --backend ollama --model qwen2.5:7b-instruct              # ทั้งชุด
```

ทั้งชุดคือ 300 abstracts × 7 เงื่อนไข × 3 seeds ราว 9,000 การเรียก LLM บน GPU T4 ควรเผื่อเวลาหลายชั่วโมง ให้ pilot ด้วย `--limit` ก่อนเสมอ (เส้นทาง Ollama ทดสอบแล้วกับ stub server เท่านั้น ยังไม่ได้ทดสอบกับ model จริง)

## ข้อควรรู้เรื่อง MockLLM

`MockLLM` ไม่ได้อ่านภาษาจริง แต่จำลองการอ่านด้วยความแม่นยำที่กำหนดไว้ใน `MockLLM.ACC`, `EXAMPLE_BOOST`, `CRITIC_ECHO` และ `COMPLY` ผลจึงสะท้อนสมมติฐานที่ใส่ไว้ ใช้สำหรับตรวจว่า pipeline, การบันทึก และการวิเคราะห์ทำงานถูก และใช้สอนการตีความผลเท่านั้น ห้ามอ้างเป็นผลวิจัย

ให้นักศึกษาลองปรับค่าเหล่านี้ (เช่น ลด `CRITIC_ECHO` เป็น 0) แล้วดูว่าข้อสรุปเปลี่ยนอย่างไร เป็นแบบฝึกหัด sensitivity analysis ที่ดี
