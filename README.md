# LoRA 기반 한국어 구어체 페르소나 파인튜닝 프로젝트
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Hub-orange)](https://huggingface.co/gyumin040911/llama-3-my-kakaotalk-lora)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
> **시계열 기반 멀티턴 데이터 전처리 파이프라인과 하이퍼파라미터 정밀 조율을 통해, 친한 친구의 카카오톡 대화 텐션을 효과적으로 재현한 LLM 파인튜닝 프로젝트입니다.**

<br>

## 1. 프로젝트 개요
* **목적**: 기존 상용 LLM(예: ChatGPT)이 구현하기 어려운 한국어 구어체 특유의 날것 느낌, 단답형, 자모음 표현(ㅋㅋ, ㅄ) 및 20대 대학생 페르소나 구축
* **핵심 성과**:
  * 제한된 GPU 자원(Colab) 환경에서 최신 가속화 엔진을 활용한 고효율 학습 성공
  * 생성 파라미터 최적화를 통해 LLM의 고질적인 병폐인 **'텍스트 반복 루프(Repetition)' 및 '문맥 이탈' 현상 해결**
  * 실제 대화 흐름을 반영한 시계열 기반 멀티턴 데이터 정제 프로세스 수립

<br>

## 2. 핵심 엔지니어링 포인트 (Problem Solving)

### ① 데이터 전처리 파이프라인 모듈화 (src/preprocess/)
연속된 카카오톡 대화 스트림을 그대로 학습할 경우, 시간 개념이 소실되어 대화 문맥이 뒤섞이는 문제가 발생했습니다. 이를 해결하기 위해 역할과 책임을 분리한 **3단계 전처리 파이프라인**을 구축했습니다.
* **0단계: Excel 기반 1차 정성적 필터링 (Human-in-the-loop)**
  * 카카오톡 원본 데이터 중 모델 학습에 부적절한 지나치게 사적인 대화, 혹은 맥락이 아예 없는 단발성 노이즈 행을 Excel을 통해 직접 눈으로 검수하며 1차 선별 및 필터링했습니다.
1. **`1_cleaner.py`**: 카카오톡 원본에 있는 개인정보를 정규식(`re`)으로 일괄 클렌징하고 행별 시간 데이터를 추가하였습니다.
2. **`2_session_splitter.py`**: **본 프로젝트의 핵심 전처리 단계.** 대화 간 공백이 **4시간 이상** 발생한 지점을 기준으로 대화 세션(Multi-turn Session)을 완전히 분리하는 시계열 데이터 가공 알고리즘을 적용하여 어제와 오늘의 문맥 엉킴을 방지했습니다.
3. **`3_formatter.py`**: 카카오톡 내보내기 원본에서 발생하는 시스템 메시지, 사진/동영상/이모티콘 태그 등을 클렌징하여 노이즈를 최소화하고 세션별로 정렬된 대화 데이터를 Hugging Face `SFTTrainer`가 즉시 학습할 수 있는 `User-Assistant` 구조의 `JSONL` 포맷으로 변환했습니다.

### ② LoRA 아키텍처 및 하이퍼파라미터 고도화 ($r=32$)
초기 기본 세팅($r=8, 16$)에서는 Base 모델 고유의 정중한 문어체 말투가 잔존하는 현상이 발생했습니다. 한국어 구어체 특유의 미묘한 뉘앙스와 초성 감성을 정밀하게 흡수하기 위해 가중치 표현 두께를 **$r=32, \alpha=64$**로 대폭 확장하여 페르소나의 밀도를 극대화했습니다.

### ③ 추론 뇌절 방지를 위한 생성 전략 최적화 (Generation Strategy)
초기 추론 시 `repetition_penalty` 설정의 부재로 인해 특정 욕설 자음 무한 연타 및 시간 표현(`10시 11시 12시...`) 감옥에 갇히는 루프 현상이 발견되었습니다. 
수많은 테스트 대화를 거쳐 **인간의 대화 리듬감과 가장 유사하면서도 뇌절을 차단하는 최적의 생성 옵션 황금 밸런스**를 도출했습니다.

```python
# src/inference.py에 적용된 최종 최적화 Generation Config
outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=64,
    temperature=0.53,
    top_p=0.85,
    top_k=50,
    repetition_penalty=1.15,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)
```

<br>

## 3. 기술 스택 및 개발 환경 (Tech Stack & Environment)
- **Base Model**: [unsloth/llama-3-8b-Instruct-bnb-4bit]
- **Fine-Tuning**: PyTorch, Hugging Face transformers, peft (LoRA), trl (SFTTrainer)
- **Memory Optimization**: Unsloth Engine, xformers (Memory-efficient Attention 적용)
- **Data Engineering**: Python, Pandas
- **Hardware**: Google Colab 환경 (V100 / A100 GPU 자원 최적화 활용)

<br>

## 4. 실제 대화 예시 (Inference Result)
| User (나) | Assistant (파인튜닝 모델) |
| :--- | :--- |
| 야 오늘 저녁에 뭐하냐? 할 거 없으면 나올? | **없음** |
| 그럼 저녁에 피자나 한 판 때릴까? 내가 삼 | **ㅇㅋ** |
| 7시에 저녁 먹기로 하고, 6시 반쯤에 저번에 만난 거기서 볼까? | **왜** |
| 아니 왜는 왜야 ㅋㅋㅋ 거기서 만나야 피자집 가기 편하니까 그렇지. 싫음? | **몰라** |
| 뭐가 몰라야 ㅋㅋㅋ 그럼 6시 반에 거기로 나오는 걸로 안다? 늦지 마라 | **ㅇㅇ** |

인사이트: repetition_penalty=1.15 도입 후 똑같은 욕설을 무의미하게 반복하지 않으며, 전후 대화의 상호 맥락을 명확히 인지하고 받아치는 고도의 페르소나 유지력을 증명했습니다.

<br>

## 5. 디렉터리 구조 (Directory Structure)
```text
My-Kakaotalk-LLM-Finetuning/
│
├── data/
│   └── .gitkeep              # (카톡 원본 데이터는 보안상 업로드에서 제외)
│
├── src/
│   ├── preprocess/        # 3단계 전처리 모듈 폴더
│   │   ├── __init__.py
│   │   ├── 1_cleaner.py          
│   │   ├── 2_session_splitter.py 
│   │   └── 3_formatter.py        
│   ├── train.py              # 로컬 및 서버 환경 범용 LoRA 파인튜닝 스크립트
│   └── inference.py          # 황금 밸런스 파라미터가 장착된 실시간 추론 스크립트
│
├── .gitignore                # 대용량 모델 파일 및 데이터 보안 설정
├── requirements.txt          # 개발 환경 재현을 위한 핵심 라이브러리 목록
└── README.md                 # 프로젝트 가이드라인
```

<br>

## 6. 실행 및 환경 구축 (Installation & Usage)
### 1) 의존성 라이브러리 설치
```Bash
git clone [https://github.com/](https://github.com/)[gyumin04]/My-Kakaotalk-LLM-Finetuning.git
cd My-Kakaotalk-LLM-Finetuning

# 핵심 딥러닝 및 Hugging Face 생태계 패키지 설치
pip install -r requirements.txt
```

### 2) LLM 가속화 엔진 (Unsloth) 단독 설치
구글 코랩 환경 또는 대규모 파인튜닝 시 VRAM 절감 및 학습 속도 가속화를 위해 의존성 충돌이 없는 단독 가속 패키지를 설치합니다.
```Bash
pip install --no-deps "unsloth[colab-new] @ git+[https://github.com/unslothai/unsloth.git](https://github.com/unslothai/unsloth.git)"
pip install unsloth_zoo
```

### 3) 실시간 추론 테스트 실행
본 프로젝트에서 최종 최적화된 LoRA 가중치는 [Hugging Face Model Hub](https://huggingface.co/gyumin040911/llama-3-my-kakaotalk-lora)에 배포되어 있습니다. `src/inference.py` 실행 시 자동으로 허깅페이스에서 가중치를 다운로드하여 빌드됩니다.
```Bash
python src/inference.py
```
