#!/bin/sh
# Strawberry Perl's MinGW g++ 13 and its static OpenSSL.
cd "$(dirname "$0")"
g++ -std=c++20 -O2 -static -Wall reward_quoter.cpp -o reward_quoter.exe \
    -I/d/Strawberry/c/include -L/d/Strawberry/c/lib -lssl -lcrypto -lws2_32 -lcrypt32 -luser32 -ladvapi32
