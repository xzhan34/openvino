mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
cmake --build . --config RelWithDebInfo --verbose -j16
cd ..
