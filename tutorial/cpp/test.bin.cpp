/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cassert>

#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>

using namespace faiss;

bool file_exist(const std::string& path) {
    return false;
    std::ifstream f(path.c_str());
    return f.good();
}

std::vector<float>
generate_vector(int64_t n, int64_t d) {
    /* 100 million limitation */
    const int64_t nlimit = 100 * 1000 * 1000; 
    const int64_t dlimit = 128; 
    assert(n < nlimit);
    assert(d < dlimit);

    std::vector<float> vec(n*d);

    for(int64_t i = 0; i < n; i++)
        for(int64_t j = 0; j < d; j++)
            vec[d * i + j] = drand48();

    return vec;
}

void
load_vector(std::vector<float>& vec, const std::string& path) {
    std::cout << "load vector from " << path << std::endl;
    std::ifstream in(path.c_str(), std::ifstream::binary);

    in.seekg (0, in.end);
    auto length = in.tellg();
    in.seekg (0, in.beg);

    vec.resize(length / sizeof(float));

    in.read((char*)vec.data(), length);
}

void
save_vector(std::vector<float>& vec, const std::string& path) {
    std::cout << "save vector to " << path << std::endl;
    std::ofstream out(path.c_str(), std::ios::binary | std::ios::out);
    out.write((char*)vec.data(), vec.size() * sizeof(float));
    out.close();
}

void
save_result(std::pair<std::vector<float>, std::vector<long>> result, int64_t topk, const std::string& path) {
    std::cout << "save result to " << path << std::endl;
    std::ofstream out(path.c_str(), std::ios::out);

    auto dists = std::get<0>(result);
    auto ids = std::get<1>(result);
    assert(ids.size() % topk == 0);
    auto nq = ids.size() / topk;
    for (size_t i = 0; i < nq; ++i) {
        for (size_t j = 0; j < topk; ++j) {
            out << ids[i * topk + j] << "\t" << dists[i * topk + j] << std::endl; 
        } 
        out << "==================" << std::endl; 
    }
    out.close();
}

class TestIndex {
 public:
     virtual void
     Load(const std::string& path) = 0;

     virtual void
     Save(const std::string& path) = 0;
     
     virtual void
     Create(std::vector<float>& vec) = 0;

     virtual std::pair<std::vector<float>, std::vector<long>>
     Search(std::vector<float>& queries, int64_t topk, int64_t nprobe) = 0;
};

class CPUTestIndex : public TestIndex {
 public:
     CPUTestIndex(faiss::gpu::StandardGpuResources& res, int64_t d, int64_t nlist)
         : res_(res),
           d_(d),
           nlist_(nlist) {
            quantizer_ = new faiss::IndexFlatL2(d);
           }

     void
     Load(const std::string& path) override {
         std::cout << "load index from " << path << std::endl;
         index_ = static_cast<faiss::IndexIVFFlat*>(read_index(path.c_str()));
     }

     void
     Save(const std::string& path) override {
         std::cout << "save index to " << path << std::endl;
         write_index(index_, path.c_str());
     }

     void
     Create(std::vector<float>& vec) override {
         std::cout << "create index" << std::endl;
         index_ = new faiss::IndexIVFFlat(quantizer_, d_, nlist_, faiss::METRIC_L2);
         index_->train(vec.size() / d_, vec.data());
         index_->add(vec.size() / d_, vec.data());
     }

     std::pair<std::vector<float>, std::vector<long>>
     Search(std::vector<float>& queries, int64_t topk, int64_t nprobe) override {
         auto nq = queries.size() / d_;

         std::vector<float> distances;
         distances.resize(nq * topk);

         std::vector<long> ids;
         ids.resize(nq * topk);

         index_->nprobe = nprobe;
         index_->search(nq, queries.data(), topk, distances.data(), ids.data());

         return std::make_pair(distances, ids);
     }

 private:
     faiss::gpu::StandardGpuResources &res_;
     faiss::IndexFlatL2* quantizer_ = nullptr;
     faiss::IndexIVFFlat *index_ = nullptr;

     int64_t d_ = 0;
     int64_t nlist_ = 0;
};


class GPUTestIndex : public TestIndex {
 public:
     GPUTestIndex(faiss::gpu::StandardGpuResources& res, int64_t d, int64_t nlist) 
         : res_(res),
           d_(d),
           nlist_(nlist) {}

     void
     Load(const std::string& path) override {
         std::cout << "load index from " << path << std::endl;
         auto cpu_index = read_index(path.c_str());
         index_ = static_cast<faiss::gpu::GpuIndexIVFFlat *>(index_cpu_to_gpu(&res_, 0, cpu_index, nullptr));
     } 

     void
     Save(const std::string& path) override {
         std::cout << "save index to " << path << std::endl;
         auto cpu_index = index_gpu_to_cpu(index_);
         write_index(static_cast<Index *>(cpu_index), path.c_str());
     }

     void
     Create(std::vector<float>& vec) override {
         std::cout << "create index" << std::endl;
         index_ = new faiss::gpu::GpuIndexIVFFlat(&res_, d_, nlist_, faiss::METRIC_L2);
         index_->train(vec.size() / d_, vec.data());
         index_->add(vec.size() / d_, vec.data());
     }

     std::pair<std::vector<float>, std::vector<long>>
     Search(std::vector<float>& queries, int64_t topk, int64_t nprobe) override {
         auto nq = queries.size() / d_;

         std::vector<float> distances;
         distances.resize(nq * topk);

         std::vector<long> ids;
         ids.resize(nq * topk);

         index_->nprobe = nprobe;
         index_->search(nq, queries.data(), topk, distances.data(), ids.data());

         return std::make_pair(distances, ids);
     }

 private:
     faiss::gpu::GpuIndexIVFFlat *index_ = nullptr;
     faiss::gpu::StandardGpuResources &res_;

     int64_t d_ = 0;
     int64_t nlist_ = 0;
};









const int64_t nb = 30000;
const int64_t nq = 2;
const int64_t d = 32;
const int64_t nlist = 2;


template<typename T>
void
test(const std::string& index_file, 
        const std::string& result_file, 
        faiss::gpu::StandardGpuResources &res, 
        std::vector<float>& base, 
        std::vector<float>& queries, 
        int64_t topk) {
    TestIndex *gpu = new T(res, d, nlist);

    if (file_exist(index_file)) {
        gpu->Load(index_file);
    } else {
        gpu->Create(base);
        gpu->Save(index_file);
    }

    auto results = gpu->Search(queries, topk, nlist);

    auto I = std::get<1>(results);
    auto D = std::get<0>(results);

    save_result(results, topk, result_file);
}


int main() {
#if CUDA_VERSION > 9000
    std::cout << "cuda >9000" << std::endl;
    std::cout << "CUDA_VERSION: " << std::to_string(CUDA_VERSION) << std::endl;
#else
    std::cout << "cuda <9000" << std::endl;
#endif

    srand48(time(0));


    faiss::gpu::StandardGpuResources res;
    
    std::vector<float> queries;
    {
        auto query_file = "query.data";
        if (file_exist(query_file)) {
            load_vector(queries, query_file);
        } else {
            queries = generate_vector(nq, d);
            save_vector(queries, query_file);
        }
    }
    

    std::vector<float> base;
    {
        auto base_file = "base.data";
        if (file_exist(base_file)) {
            load_vector(base, base_file);
        } else {
            base = generate_vector(nb, d);
            save_vector(base, base_file);
        }
    }

    auto kkkkkk = 10000;

    test<GPUTestIndex>("gpu-index.data", "gpu-result.data", res, base, queries, kkkkkk);
    test<CPUTestIndex>("cpu-index.data", "cpu-result.data", res, base, queries, kkkkkk);

    return 0;
}

