1차 업무 안내

[수행 방법]

1. Git 설정 및 저장소 생성

먼저, GitHub 또는 GitLab에 새로운 저장소(repository)를 생성하고 로컬 환경과 연결

1.1 GitHub에서 새로운 저장소 생성
GitHub에 로그인하고 새 저장소(New Repository)를 만든다.
저장소 이름을 설정하고 "초기화 안 함(No README, No .gitignore 선택)" 상태로 둔다.
생성된 저장소 URL을 복사한다.

1.2 로컬 환경에서 Git 초기화 & 연동
# Git 초기화
> git init
# Github 저장소 복제
> git clone [저장소 URL]
# 원격 저장소의 내용을 로컬로 가져오기 (있을 경우)
> git pull origin main

2. Branch 및 Commit 실습
업무를 진행할 새로운 브랜치를 만들고, 변경 사항을 관리

# [브랜치 이름] 브랜치 생성
> git branch [브랜치 이름]
# 생성한 브랜치로 이동
> git checkout [브랜치 이름]

(첨부: 원본 슬라이드 내용 요약)
