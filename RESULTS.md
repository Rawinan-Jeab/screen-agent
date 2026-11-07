# ผลการทดสอบ ScreenAgent Prototype

รันเมื่อ 7 พฤศจิกายน 2026 · backend: **MockLLM** (จำลอง) · test 300 abstracts (เกี่ยวข้อง 150) · 3 seeds ต่อเงื่อนไข, majority vote

> ตัวเลขทั้งหมดมาจาก MockLLM ซึ่งจำลองการอ่านด้วยความแม่นยำที่ผู้พัฒนากำหนด ผลจึงยืนยันได้ว่า code ทำงานถูกต้องและการวิเคราะห์ใช้งานได้ แต่ไม่ใช่หลักฐานว่า agent ดีกว่าจริง ต้องรันซ้ำกับ local model จริงก่อนนำไปเขียน paper

## 1. Unit test ของแต่ละ component: ผ่าน 23/23

| Component | จำนวน test | สิ่งที่ตรวจ | ผล |
| --- | --- | --- | --- |
| Screener | 3 | ลำดับ ReAct (get_criteria → search_examples → ตัดสิน), ครอบ abstract ด้วย `<untrusted_abstract>`, ปิด memory แล้วไม่ค้นตัวอย่าง | ผ่าน |
| Critic | 4 | เห็นต่างในเกณฑ์ที่ทำให้ exclude → uncertain, เห็นตรงกัน → exclude, ค้าน include → uncertain, prompt มีคำตัดสินของ Screener | ผ่าน |
| Supervisor | 4 | บันทึกคำตัดสิน, exclusion ที่ confidence ต่ำ → ส่งมนุษย์, ไม่เกิน step budget, ล้าง working memory ทุก abstract | ผ่าน |
| Permission gate | 4 | low / medium (task-bound) / high (ต้องอนุมัติ) / tool ที่ไม่รู้จัก, injected action ถูกบล็อก end-to-end, ไม่มี gate แล้วการโจมตีสำเร็จ | ผ่าน |
| Input guard | 2 | ตรวจพบ payload ครบ 3 แบบ, ไม่มี false positive บน abstract ปกติ 370 เรื่อง | ผ่าน |
| Memory | 4 | semantic memory ปฏิเสธ test example, retrieval ได้เฉพาะ dev, episodic เขียนได้เฉพาะมนุษย์, procedural ตรวจจับการแก้หลัง freeze | ผ่าน |
| Audit log | 1 | hash chain ถูกต้อง และตรวจจับการแก้ record ได้ | ผ่าน |
| Baseline | 1 | keyword baseline คืน label ถูกรูปแบบ | ผ่าน |

ผลดิบอยู่ใน `results/unit_tests.txt`

## 2. ผลการทดลองตาม Experiment Plan

| เงื่อนไข | Recall | 95% CI | Precision | WSS | ส่งมนุษย์ (uncertain) | Token/abstract | Attack success |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B0 keyword | 0.293 | 0.226–0.371 | 0.786 | 0.107 | 0% | 0 | – |
| B1 single-call | 0.673 | 0.595–0.743 | 0.944 | 0.317 | 0% | 320 | – |
| B2 retrieval | 0.760 | 0.686–0.821 | 0.934 | 0.353 | 0% | 789 | – |
| **ScreenAgent (เต็ม)** | **1.000** | 0.975–1.000 | 0.794 | 0.370 | 27.9% | 1,457 | **0/30** |
| − critic | 0.967 | 0.924–0.986 | 0.843 | 0.393 | 12.8% | 935 | 2/30 |
| − gate | 1.000 | 0.975–1.000 | 0.794 | 0.370 | 27.9% | 1,457 | 6/30 |
| − memory | 0.973 | 0.933–0.990 | 0.745 | 0.320 | 32.4% | 992 | 0/30 |

การทดสอบสมมติฐาน (McNemar exact แบบ paired บน abstract ที่เกี่ยวข้อง 150 เรื่อง, Holm สำหรับ 4 การเปรียบเทียบ)

| การเปรียบเทียบ | ระบบเต็มถูกฝ่ายเดียว | อีกฝ่ายถูกฝ่ายเดียว | Holm p | Cohen's h | สรุป |
| --- | --- | --- | --- | --- | --- |
| เต็ม vs B1 | 49 | 0 | < .001 | 1.22 | – |
| เต็ม vs B2 (H1) | 36 | 0 | < .001 | 1.02 | H1 ได้รับการสนับสนุน |
| เต็ม vs − critic (H2) | 5 | 0 | .125 | 0.37 | H2 ยังไม่มีนัยสำคัญ |
| เต็ม vs − gate (H3, recall) | 0 | 0 | 1.00 | 0.00 | gate ไม่ลด recall |
| เต็ม vs − memory (H4, exploratory) | 4 | 0 | .125 (ไม่ปรับ) | – | ยังไม่มีนัยสำคัญ |

H3 ด้าน safety: attack success ลดจาก 6/30 เหลือ 0/30 (McNemar exact บน 6 คู่ที่ต่างกัน p ≈ .031)

## 3. ผลของ agent แต่ละตัว

**Screener:** ความแม่นยำรายเกณฑ์บน abstract ปกติ คือ 0.972 สำหรับข้อความชัดเจน และ 0.919 สำหรับข้อความกำกวม เมื่อปิด memory ความแม่นยำบนข้อความกำกวมลดเหลือ 0.834 ส่วนข้อความชัดเจนไม่เปลี่ยน แปลว่า memory ช่วยเฉพาะกรณียาก

**Critic:** จับข้อผิดพลาดของ Screener บน abstract ที่เกี่ยวข้องได้ 49 จาก 87 ครั้ง (catch rate 56%) และค้านผิดเพียง 5% ของกรณีที่ Screener ถูก ข้อผิดพลาดที่หลุดส่วนใหญ่มาจากการที่ Critic เห็นด้วยตาม Screener โดยไม่ตรวจเอง (ตั้งไว้ 40% ใน mock) ซึ่งตรงกับ failure mode ใน MAST taxonomy ที่สอนใน Module 3

**Supervisor:** จาก 900 การตัดสิน (300 × 3 seeds) เป็น include 321, exclude 328, uncertain 251 โดย uncertain มาจาก Critic ค้านเกณฑ์ที่ทำให้ exclude 109 ครั้ง, Critic ค้าน include 76 ครั้ง และ exclusion ที่ confidence ต่ำ 66 ครั้ง ใช้เฉลี่ย 6.3 steps ต่อ abstract (สูงสุด 8) ไม่มีครั้งไหนเกิน step budget

**Permission gate:** บล็อก export_results ที่ถูก injection สั่ง 8 ครั้ง และบล็อก record_decision ที่พยายามแก้ abstract อื่น 11 ครั้ง เมื่อไม่มี gate คำสั่งทั้ง 19 ครั้งนี้ถูกทำจริง

**Input guard:** ตรวจพบ abstract ที่ฝัง injection ครบ 100% และไม่มี false positive บน abstract ปกติ (แต่ guard เป็นแค่ pattern matching ผู้โจมตีที่เขียนต่างออกไปเลี่ยงได้ จึงต้องมี gate เป็นชั้นถัดไป)

**Memory:** retrieval ไม่เคยดึง abstract จาก test (0 ครั้ง) และตัวอย่าง top-5 มี label ตรงกับ abstract ที่ถามเพียง 56% แปลว่าประโยชน์ของ memory มาจากการเห็นสำนวนที่คล้ายกัน ไม่ใช่จาก label

**Audit log:** ระบบเต็ม seed 1 บันทึก 1,305 records และ hash chain ถูกต้องทั้งสาย

## 4. ข้อสังเกตสำหรับการตีความ (ใช้สอนได้)

1. **Recall ที่สูงขึ้นมีต้นทุนเป็นงานของมนุษย์** ScreenAgent ส่งมนุษย์ตรวจ 27.9% ของ abstract ขณะที่ baseline ไม่ส่งเลย เมื่อดู WSS ซึ่งรวมงานที่มนุษย์ยังต้องอ่าน ระบบเต็ม (0.370) ประหยัดงานได้ใกล้เคียง B2 (0.353) ดังนั้นข้อสรุปที่ถูกต้องคือ "recall สูงขึ้นมากโดยงานรวมไม่เพิ่ม" ไม่ใช่ "ประหยัดงานได้มากกว่า"
2. **ใช้ token มากกว่า B2 ราว 1.8 เท่า** ต้องรายงานคู่กับ recall เสมอ
3. **H2 ไม่ผ่านเพราะ recall ชนเพดาน** มีคู่ที่ต่างกันเพียง 5 คู่ ควรเพิ่มจำนวน abstract ที่เกี่ยวข้อง หรือใช้ข้อมูลที่ยากขึ้น จากความต่างที่เห็น (B2 เทียบระบบเต็ม) sample size แบบ paired ที่ต้องใช้คือราว 31 abstract ที่เกี่ยวข้อง แต่สำหรับ H2 ต้องมากกว่านี้มาก
4. **Precision ลดลง** (0.794 เทียบ 0.934 ของ B2) เพราะ uncertain ถูกนับเป็น "ไม่ exclude" ต้องอธิบายใน Discussion
5. **ผลทั้งหมดขึ้นกับค่าที่ตั้งใน MockLLM** เช่น ถ้าไม่ตั้ง `EXAMPLE_BOOST` ไว้ memory จะไม่มีผลเลย นี่คือเหตุผลที่ต้องรันกับ model จริง

## 5. ไฟล์ผลลัพธ์

- `results/tables.md` ตารางผลที่สร้างอัตโนมัติ
- `results/results.json` ตัวเลขทุกตัว รวมผลราย agent
- `results/runs.jsonl` ทุก run พร้อม trace ของ tool call (6,300 แถว)
- `results/audit_full_seed1.jsonl` audit log แบบ hash chain
- `results/corpus.jsonl` ชุด abstract สังเคราะห์พร้อม ground truth
- `results/unit_tests.txt` ผล unit test
