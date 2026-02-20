# Notion 업로드 자동화 런북

작성일: 2026-02-20

## 목적
- 매번 수동으로 올리지 않고, `업로드 블록 md` 기반으로 노션 페이지 생성 + 파일 첨부를 자동화한다.

## 1) 준비
```bash
export NOTION_TOKEN="ntn_..."
```

## 2) 실행 (기본: 2026년 > 2월 > 일주일 자동 탐색)
```bash
python3 unitree_go2_phm/scripts/rsl_rl/reports/upload_to_notion.py \
  --upload-md unitree_go2_phm/scripts/rsl_rl/reports/2026-02-20_notion_upload_block.md \
  --page-title "2026-02-20 새벽 실험 업데이트 (TB+영상)" \
  --report-md unitree_go2_phm/scripts/rsl_rl/reports/2026-02-20_realobs_growth_journal.md
```

## 2-A) 하위 페이지 만들지 않고 `일주일` 본문에 직접 붙이기 (권장)
```bash
python3 unitree_go2_phm/scripts/rsl_rl/reports/upload_to_notion.py \
  --append-to-week-page \
  --week-page-id 30cbdb9f-64f2-810d-9483-eb178a729b9c \
  --upload-md unitree_go2_phm/scripts/rsl_rl/reports/2026-02-week_daily_notion_upload.md \
  --page-title "2026-02-20 학습 완료 업데이트 최종본 (데일리 스토리 + TB + Walking 영상)" \
  --report-md unitree_go2_phm/scripts/rsl_rl/reports/2026-02-20_latest_runs_growth_journal.md
```

## 3) 특정 주간 페이지 ID를 직접 지정하고 싶을 때
```bash
python3 unitree_go2_phm/scripts/rsl_rl/reports/upload_to_notion.py \
  --week-page-id 30cbdb9f-64f2-810d-9483-eb178a729b9c \
  --upload-md unitree_go2_phm/scripts/rsl_rl/reports/2026-02-20_notion_upload_block.md \
  --page-title "2026-02-20 새벽 실험 업데이트 (TB+영상)"
```

## 4) Dry-run (실제 업로드 없이 점검)
```bash
python3 unitree_go2_phm/scripts/rsl_rl/reports/upload_to_notion.py \
  --upload-md unitree_go2_phm/scripts/rsl_rl/reports/2026-02-20_notion_upload_block.md \
  --page-title "TEST" \
  --dry-run
```

## 5) 규칙
- `upload-md`에서 백틱(`...`)으로 감싼 경로를 파일로 인식해 업로드한다.
- 이미지(`png/jpg`)는 image 블록, 영상(`mp4`)은 video 블록으로 첨부한다.
- 경로가 없거나 파일이 없으면 텍스트 bullet로 남긴다.
