#!/usr/bin/env sh
git push
CURR_BRANCH=$(git rev-parse --abbrev-ref HEAD)
ssh root@107.172.159.51 /bin/sh <<EOF  
	cd /root/ric-staging
	systemctl stop ric-staging
	git pull -f
	git checkout $CURR_BRANCH
	systemctl start ric-staging
EOF
