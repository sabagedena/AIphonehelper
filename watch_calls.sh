#!/bin/sh
# Live view of the phone line: each call's number and both sides of the conversation, without the
# debug log (which also prints the whole system prompt every turn). On the server:
#   sh ~/AIphonehelper/watch_calls.sh        (last 300 log lines, then follows new calls; Ctrl+C stops)
# From Windows:
#   ssh -t root@<server> "sh ~/AIphonehelper/watch_calls.sh"
docker logs -f --tail "${1:-300}" georgian-ai-phone 2>&1 \
  | grep --line-buffered -E '^(📞|📴|📝|✋)|^🗣.* აბონენტი:|^🤖 +დამხმარე:|Caller\)'
