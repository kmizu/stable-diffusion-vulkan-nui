#!/usr/bin/env bash
# Stable Diffusion NUI launcher. Prints nothing to stdout itself (MCP stdio mode uses stdout).
set -u
cd "$(dirname "$0")" || exit 1
HOME_DIR="$(pwd)"

JAVA=""
if [ -n "${SDFX_SCALATORCH_JAVA:-}" ] && [ -x "$SDFX_SCALATORCH_JAVA" ]; then JAVA="$SDFX_SCALATORCH_JAVA"
elif [ -n "${JAVA25_HOME:-}" ] && [ -x "$JAVA25_HOME/bin/java" ]; then JAVA="$JAVA25_HOME/bin/java"
elif [ -n "${JAVA_HOME:-}" ] && [ -x "$JAVA_HOME/bin/java" ]; then
  case "$JAVA_HOME" in *25*) JAVA="$JAVA_HOME/bin/java";; esac
fi
if [ -z "$JAVA" ]; then
  for d in /usr/lib/jvm/*25* /Library/Java/JavaVirtualMachines/*25*/Contents/Home; do
    if [ -x "$d/bin/java" ]; then JAVA="$d/bin/java"; break; fi
  done
fi
if [ -z "$JAVA" ]; then
  if [ -n "${JAVA_HOME:-}" ] && [ -x "$JAVA_HOME/bin/java" ]; then JAVA="$JAVA_HOME/bin/java"
  elif command -v java >/dev/null 2>&1; then JAVA="$(command -v java)"
  else echo "[sdfx] Java 17 or newer was not found. Install a JDK 17+ and set JAVA_HOME or add java to PATH." >&2; exit 1; fi
fi

VER="$("$JAVA" -version 2>&1 | sed -n 's/.*version "\([^"]*\)".*/\1/p' | head -n1)"
MAJOR="${VER%%.*}"
case "$MAJOR" in ''|*[!0-9]*) MAJOR=0;; esac
if [ "$MAJOR" -lt 17 ]; then echo "[sdfx] Java 17 or newer is required (found \"$VER\" at $JAVA)." >&2; exit 1; fi

JAR="$HOME_DIR/target/stable-diffusion-nui.jar"
if [ ! -f "$JAR" ]; then
  echo "[sdfx] target/stable-diffusion-nui.jar is missing, building it with sbt (or run scripts/build.sh)..." >&2
  if ! command -v sbt >/dev/null 2>&1; then echo "[sdfx] sbt was not found. Install sbt, then run scripts/build.sh." >&2; exit 1; fi
  sbt -batch ui/assembly 1>&2 || { echo "[sdfx] build failed." >&2; exit 1; }
fi

for a in "$@"; do
  if [ "$a" = "--mcp" ]; then
    exec "$JAVA" -Xlog:disable -Xlog:all=warning:stderr -Djava.awt.headless=true -Dfile.encoding=UTF-8 -Xmx2g \
      -Dsdfx.home="$HOME_DIR" -cp "$JAR" sdfx.mcp.McpMain "$@"
  fi
done
exec "$JAVA" -Dprism.allowhidpi=true -Dfile.encoding=UTF-8 -Xmx2g -Dsdfx.home="$HOME_DIR" -jar "$JAR" "$@"
