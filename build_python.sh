cargo rustc --release --crate-type cdylib --features python
ln -s target/release/libquaternion.dylib ./quaternion.so
