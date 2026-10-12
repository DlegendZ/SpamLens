# Human-reviewed error analysis

Published labels and frozen models remain unchanged.

| Category | Count | FP | FN |
|---|---:|---:|---:|
| Transactional / OTP resembles promotion | 6 | 3 | 3 |
| Short or ambiguous message | 4 | 3 | 1 |
| Conversational spam / romance / job scam | 5 | 0 | 5 |
| Obfuscated wording | 2 | 0 | 2 |
| Non-English or code-mixed | 2 | 1 | 1 |
| Probable published-label noise | 5 | 2 | 3 |

## Team-anonymized examples

**Transactional / OTP resembles promotion:** delivery due tomorrow Friday [NUM] Jun [NUM]. if not home we'll leave it in a safe place See

**Transactional / OTP resembles promotion:** Order Confirmed Hi [NAME]! Your order is confirmed. We will deliver it between [NUM] - [NUM]. Track your order on the app at [URL]

**Short or ambiguous message:** [NUM]/[NUM] [NUM]tf[NUM]p

**Short or ambiguous message:** Indian Team Penalized by Animal Activists for Hunting [NUM] Kangaroos on thursday?

**Conversational spam / romance / job scam:** What is the color of your eyes and hair? Please write back, I'm eagerly waiting for your response.

**Conversational spam / romance / job scam:** Hello I'm [NAME] be you [NAME] I'm not sure if I get the phone number wrong I make the wrong call and I'm so sorry

**Obfuscated wording:** | am [NAME]. | tried to call and left a voicemail concerning biz financing. Give me a ring back to discuss details.

**Obfuscated wording:** Call Us For [NAME] It can Make u Happy [PHONE] Near Your Location [NAME]

**Non-English or code-mixed:** Launching New Tower i [NAME] Centrum Sion[NUM]/[NUM] BHK Starting [MONEY]

**Non-English or code-mixed:** Can apply for lol account transfer liao! Got free skins b4 [NUM] jan

**Probable published-label noise:** TBS/PERSOLVO. been chasing us since Sept for [MONEY] definitely not paying now thanks to your information. We will ignore them. [NAME]. Manchester.

**Probable published-label noise:** Valentines offer! Recharge your MTS Mobile/Data card online and avail discounts on Flowers, Soft toys and much more. Offer valid on online recharge above [MONEY]


Train-only suspect labels reviewed: 5. Suspect flags are diagnostics, not confirmed ground-truth errors.
