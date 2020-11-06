#/bin/sh
mkdir "build_tmp"

git clone -b countries git@github.com:sust-cs-uob/eam-data-tools.git build_tmp/eam-data-tools
git clone -b countries git@github.com:sust-cs-uob/eam-core.git build_tmp/eam-core

cp build_tmp/eam-core/etc/Dockerfile build_tmp

cd build_tmp
docker build --pull --rm -f "Dockerfile" -t eam-core:latest "eam-core"
