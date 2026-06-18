# PXL Explicit Prefetch Test

PXL 디바이스 메모리에서 explicit device prefetch(`pxl::prefetchMemory`)가
cold read 성능에 주는 효과를 측정하는 벤치마크 모드입니다.

- **Repo:** `xcena-dev/mio`
- **Branch:** `feature/pxl-explicit-prefetch`

## `prefetch_test` 모드란

워킹셋 W에 대해 **2×W**(region A + region B)를 할당하고 세 단계로 동작합니다:

1. region A에 sequential **write**
2. region B를 sequential **read** → region A를 캐시에서 축출(cold화)
3. region A를 cold **read** 하면서 `pxl::prefetchMemory`로 N chunk 앞을 prefetch

③번 read 대역폭이 결과값이라, explicit prefetch 효과를 볼 수 있습니다.

## 빌드

```bash
cd mio
./build.sh        # 의존성 설치 + msr 모듈 로드 + 빌드
# libpxl + pxl/memory.hpp 필요 (없으면 빌드 에러)
```

## 실행 예제

워킹셋 1 TiB, 4 threads, HW prefetcher OFF:

```bash
# baseline (prefetch 없음)
sudo ./build/microbench --mode prefetch_test --threads 4 \
  --memory-per-thread 262144 --pxl-device 0 --prefetch OFF \
  --explicit-prefetch-ahead 0

# prefetch ahead=100 (chunk 32 MiB → 약 3.2 GiB 앞을 prefetch)
sudo ./build/microbench --mode prefetch_test --threads 4 \
  --memory-per-thread 262144 --pxl-device 0 --prefetch OFF \
  --explicit-prefetch-ahead 100
```

## 주요 옵션

| 옵션 | 설명 |
|------|------|
| `--explicit-prefetch-ahead N` | N chunk 앞을 prefetch (0 = baseline, prefetch 없음) |
| `--prefetch-chunk-size <bytes>` | prefetch 단위 (기본 32 MiB) |
| `--pxl-device <id>` | PXL 디바이스 (prefetch_test 필수) |
