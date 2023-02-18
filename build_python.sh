# note: symbolic link is for conveninence and "dylib" extension is for mac
# will need to be modified for other OS
cargo rustc --release --crate-type cdylib --features python
ln -s target/release/libquaternion.dylib ./quaternion.so
