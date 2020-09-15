# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

-include ../../makefile.inc

all: test.bin
	./test.bin

test.bin: test.bin.cpp ../../libfaiss.a
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) -I../../.. $(LIBS)

clean:
	rm -f test.bin
	rm *.data

.PHONY: all clean
