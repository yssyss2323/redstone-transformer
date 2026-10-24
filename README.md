# 레드스톤 트랜스포머 — 0단계: 학습 + 정수 레퍼런스

서로 다른 숫자 4개(0~7)를 정렬하는 1층 1헤드 디코더 온리 트랜스포머를,
레드스톤 하드웨어 규칙(2의 거듭제곱 스케일, int4 가중치, 반올림 시프트, 지수 ROM softmax)으로 학습합니다.

## 실행 순서 (Colab 또는 로컬, CPU로 충분)

```bash
pip install torch          # Colab은 이미 설치되어 있음

# 1) 한 조합 학습 + weights.json 저장 (CPU 기준 1~3분)
python train.py --d 8 --ffn 16 --export

# 2) 정수 레퍼런스 검증 + 예시 덤프
python int_ref.py
python int_ref.py 3 6 1 4
python int_ref.py --no-cache 3 6 1 4

# 3) (선택) d_model / FFN 크기 비교. 12회 학습이라 CPU로 20~30분
python train.py --sweep
```

## 확인할 것

| 체크 | 기대값 |
|---|---|
| `train.py` 마지막 줄 `int_test` | 1.0 (테스트 336개 전부 정답) |
| `int_ref.py`의 `정렬 정답` | 1680/1680 |
| `int_ref.py`의 `PyTorch QAT 일치` | 1680/1680 (하나라도 다르면 정수 규칙 재현이 틀린 것) |
| attention 히스토리 | 처음 3개 출력은 정답 숫자 위치에 `@@`, 마지막 출력은 퍼지거나 자기 자신을 봄 |
| KV 캐시 on/off | 8레인 연산 약 410회 vs 약 760회 |

## 스윕에서 볼 것

- `float_te`: 실수 모델 테스트 정확도
- `int_noQAT`: 보정만 하고 QAT 전 정수 정확도 (보통 크게 떨어짐)
- `int_te`: QAT 후 정수 테스트 정확도 ← **크기 결정 기준**

d_model 4는 실패, 8(FFN 16)은 100%가 확인됐고, 6은 아직 결론이 안 났습니다.
