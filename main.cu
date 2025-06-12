#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <unistd.h>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include "llmc/cuda_common.h"
#include "llmc/tokenizer.h"
#include "llmc/utils.h"

#ifdef PTX
#include "kernels_ptx.h"
#else
#include "kernels_cuda.cuh"
#endif

// ----------------------------------------------------------------------------
// GPT-2 model definition
typedef struct {
  int max_seq_len;       // max sequence length, e.g. 1024
  int vocab_size;        // vocab size, e.g. 50257
  int padded_vocab_size; // padded to e.g. %128==0, 50304
  int num_layers;        // number of layers, e.g. 12
  int num_heads;         // number of heads in attention, e.g. 12
  int channels;          // number of channels, e.g. 768
} GPT2Config;

// the parameters of the model
#define NUM_PARAMETER_TENSORS 16
typedef struct {
  float *wte;      // (V, C)
  float *wpe;      // (maxT, C)
  float *ln1w;     // (L, C)
  float *ln1b;     // (L, C)
  float *qkvw;     // (L, 3*C, C)
  float *qkvb;     // (L, 3*C)
  float *attprojw; // (L, C, C)
  float *attprojb; // (L, C)
  float *ln2w;     // (L, C)
  float *ln2b;     // (L, C)
  float *fcw;      // (L, 4*C, C)
  float *fcb;      // (L, 4*C)
  float *fcprojw;  // (L, C, 4*C)
  float *fcprojb;  // (L, C)
  float *lnfw;     // (C)
  float *lnfb;     // (C)
} ParameterTensors;

void fill_in_parameter_sizes(size_t *param_sizes, GPT2Config config) {
  size_t Vp = config.padded_vocab_size;
  size_t C = config.channels;
  size_t maxT = config.max_seq_len;
  size_t L = config.num_layers;
  param_sizes[0] = Vp * C;           // wte
  param_sizes[1] = maxT * C;         // wpe
  param_sizes[2] = L * C;            // ln1w
  param_sizes[3] = L * C;            // ln1b
  param_sizes[4] = L * (3 * C) * C;  // qkvw
  param_sizes[5] = L * (3 * C);      // qkvb
  param_sizes[6] = L * C * C;        // attprojw
  param_sizes[7] = L * C;            // attprojb
  param_sizes[8] = L * C;            // ln2w
  param_sizes[9] = L * C;            // ln2b
  param_sizes[10] = L * (4 * C) * C; // fcw
  param_sizes[11] = L * (4 * C);     // fcb
  param_sizes[12] = L * C * (4 * C); // fcprojw
  param_sizes[13] = L * C;           // fcprojb
  param_sizes[14] = C;               // lnfw
  param_sizes[15] = C;               // lnfb
}

// allocate memory for the parameters and point the individual tensors to the
// right places
float *malloc_and_point_parameters(ParameterTensors *params,
                                   size_t *param_sizes) {
  size_t num_parameters = 0;
  for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
    num_parameters += param_sizes[i];
  }
  // malloc all parameters all at once
  float *params_memory;
  cudaCheck(cudaMallocManaged(&params_memory, num_parameters * sizeof(float)));
  // assign all the tensors
  float **ptrs[] = {
      &params->wte,     &params->wpe,     &params->ln1w,     &params->ln1b,
      &params->qkvw,    &params->qkvb,    &params->attprojw, &params->attprojb,
      &params->ln2w,    &params->ln2b,    &params->fcw,      &params->fcb,
      &params->fcprojw, &params->fcprojb, &params->lnfw,     &params->lnfb};
  char *params_memory_iterator = (char *)params_memory;
  for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
    *(ptrs[i]) = (float *)params_memory_iterator;
    params_memory_iterator += param_sizes[i] * sizeof(float);
  }
  return params_memory;
}

#define NUM_ACTIVATION_TENSORS 23
typedef struct {
  float *encoded;   // (B, T, C)
  float *ln1;       // (L, B, T, C)
  float *ln1_mean;  // (L, B, T)
  float *ln1_rstd;  // (L, B, T)
  float *qkv;       // (L, B, T, 3*C)
  float *atty;      // (L, B, T, C)
  float *preatt;    // (L, B, NH, T, T)
  float *att;       // (L, B, NH, T, T)
  float *attproj;   // (L, B, T, C)
  float *residual2; // (L, B, T, C)
  float *ln2;       // (L, B, T, C)
  float *ln2_mean;  // (L, B, T)
  float *ln2_rstd;  // (L, B, T)
  float *fch;       // (L, B, T, 4*C)
  float *fch_gelu;  // (L, B, T, 4*C)
  float *fcproj;    // (L, B, T, C)
  float *residual3; // (L, B, T, C)
  float *lnf;       // (B, T, C)
  float *lnf_mean;  // (B, T)
  float *lnf_rstd;  // (B, T)
  float *logits;    // (B, T, V)
  float *probs;     // (B, T, V)
  float *losses;    // (B, T)
} ActivationTensors;

void fill_in_activation_sizes(size_t *act_sizes, GPT2Config config, int B,
                              int T) {
  size_t C = config.channels;
  size_t NH = config.num_heads;
  size_t L = config.num_layers;
  size_t Vp = config.padded_vocab_size;
  act_sizes[0] = B * T * C;          // encoded
  act_sizes[1] = L * B * T * C;      // ln1
  act_sizes[2] = L * B * T;          // ln1_mean
  act_sizes[3] = L * B * T;          // ln1_rstd
  act_sizes[4] = L * B * T * 3 * C;  // qkv
  act_sizes[5] = L * B * T * C;      // atty
  act_sizes[6] = L * B * NH * T * T; // preatt
  act_sizes[7] = L * B * NH * T * T; // att
  act_sizes[8] = L * B * T * C;      // attproj
  act_sizes[9] = L * B * T * C;      // residual2
  act_sizes[10] = L * B * T * C;     // ln2
  act_sizes[11] = L * B * T;         // ln2_mean
  act_sizes[12] = L * B * T;         // ln2_rstd
  act_sizes[13] = L * B * T * 4 * C; // fch
  act_sizes[14] = L * B * T * 4 * C; // fch_gelu
  act_sizes[15] = L * B * T * C;     // fcproj
  act_sizes[16] = L * B * T * C;     // residual3
  act_sizes[17] = B * T * C;         // lnf
  act_sizes[18] = B * T;             // lnf_mean
  act_sizes[19] = B * T;             // lnf_rstd
  act_sizes[20] = B * T * Vp;        // logits
  act_sizes[21] = B * T * Vp;        // probs
  act_sizes[22] = B * T;             // losses
}

float *malloc_and_point_activations(ActivationTensors *acts,
                                    size_t *act_sizes) {
  size_t num_activations = 0;
  for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
    num_activations += act_sizes[i];
  }
  float *acts_memory;
  cudaCheck(cudaMalloc((void **)&acts_memory, num_activations * sizeof(float)));

  cudaCheck(cudaMemset(acts_memory, 0, num_activations * sizeof(float)));
  float **ptrs[] = {
      &acts->encoded,   &acts->ln1,       &acts->ln1_mean, &acts->ln1_rstd,
      &acts->qkv,       &acts->atty,      &acts->preatt,   &acts->att,
      &acts->attproj,   &acts->residual2, &acts->ln2,      &acts->ln2_mean,
      &acts->ln2_rstd,  &acts->fch,       &acts->fch_gelu, &acts->fcproj,
      &acts->residual3, &acts->lnf,       &acts->lnf_mean, &acts->lnf_rstd,
      &acts->logits,    &acts->probs,     &acts->losses};
  float *acts_memory_iterator = acts_memory;
  for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
    *(ptrs[i]) = acts_memory_iterator;
    acts_memory_iterator += act_sizes[i];
  }
  return acts_memory;
}

typedef struct {
  GPT2Config config;
  ParameterTensors params;
  size_t param_sizes[NUM_PARAMETER_TENSORS];
  float *params_memory;
  size_t num_parameters;
  ActivationTensors acts;
  size_t act_sizes[NUM_ACTIVATION_TENSORS];
  float *acts_memory;
  size_t num_activations;
  int batch_size;
  int seq_len;
  int *inputs; // the input tokens for the current forward pass
} GPT2;

void gpt2_build_from_checkpoint(GPT2 *model, const char *checkpoint_path) {
  FILE *model_file = fopenCheck(checkpoint_path, "rb");
  int model_header[256];
  freadCheck(model_header, sizeof(int), 256, model_file);
  if (model_header[0] != 20240326) {
    printf("Bad magic model file\n");
    exit(1);
  }
  if (model_header[1] != 3) {
    printf("Bad version in model file\n");
    printf("---> HINT: try to re-run `python train_gpt2.py`\n");
    exit(1);
  }

  // read in hyperparameters
  size_t maxT, V, Vp, L, NH, C; // size_t to prevent int overflow
  model->config.max_seq_len = maxT = model_header[2];
  model->config.vocab_size = V = model_header[3];
  model->config.num_layers = L = model_header[4];
  model->config.num_heads = NH = model_header[5];
  model->config.channels = C = model_header[6];
  model->config.padded_vocab_size = Vp = model_header[7];
  printf("[GPT-2]\n");
  printf("max_seq_len: %zu\n", maxT);
  printf("vocab_size: %zu\n", V);
  printf("padded_vocab_size: %zu\n", Vp);
  printf("num_layers: %zu\n", L);
  printf("num_heads: %zu\n", NH);
  printf("channels: %zu\n", C);

  // allocate space for all the parameters and read them in
  fill_in_parameter_sizes(model->param_sizes, model->config);

  // count the number of parameters
  size_t num_parameters = 0;
  for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
    num_parameters += model->param_sizes[i];
  }
  printf("num_parameters: %zu\n", num_parameters);
  model->num_parameters = num_parameters;

  // read in all the parameters from file
  model->params_memory =
      malloc_and_point_parameters(&model->params, model->param_sizes);

  freadCheck(model->params_memory, sizeof(float), num_parameters, model_file);
  cudaCheck(cudaDeviceSynchronize());

  // other inits
  model->acts_memory = NULL;
  model->inputs = NULL;
  model->batch_size = 0;
  model->seq_len = 0;
}

void gpt2_forward(GPT2 *model, int *inputs, size_t B, size_t T) {
  // ensure the model was initialized or error out
  if (model->params_memory == NULL) {
    printf("Error: model was not initialized properly.\n");
    exit(1);
  }

  // convenience parameters (size_t to help prevent int overflow)
  size_t V = model->config.vocab_size;
  size_t Vp = model->config.padded_vocab_size;
  size_t L = model->config.num_layers;
  size_t NH = model->config.num_heads;
  size_t C = model->config.channels;

  // validate inputs, all indices must be in the range [0, V)
  for (int i = 0; i < B * T; i++) {
    assert(0 <= inputs[i] && inputs[i] < V);
  }

  // allocate space for all the activations if needed (done here, lazily)
  if (model->acts_memory == NULL) {
    // record the current B,T as well
    model->batch_size = B;
    model->seq_len = T;
    // and now allocate the space
    fill_in_activation_sizes(model->act_sizes, model->config, B, T);
    size_t num_activations = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
      num_activations += model->act_sizes[i];
    }
    printf("num_activations: %zu\n", num_activations);
    model->num_activations = num_activations;
    model->acts_memory =
        malloc_and_point_activations(&model->acts, model->act_sizes);
    // also create memory for caching inputs
    // model->inputs = (int *)mallocCheck(B * T * sizeof(int));
    cudaCheck(cudaMalloc((void **)&model->inputs, B * T * sizeof(int)));
  } else {
    // validate B,T is consistent with how we've allocated the memory before
    // in principle we could get more clever here in the future, for now this is
    // safest
    if (B != model->batch_size || T != model->seq_len) {
      printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size,
             model->seq_len, (int)B, (int)T);
      exit(EXIT_FAILURE);
    }
  }

  // cache the inputs
  // memcpy(model->inputs, inputs, B * T * sizeof(int));
  cudaCheck(cudaMemcpy(model->inputs, inputs, B * T * sizeof(int),
                       cudaMemcpyHostToDevice));

  // forward pass
  ParameterTensors params = model->params; // for brevity
  ActivationTensors acts = model->acts;
  float *residual;
  encoder_forward(acts.encoded, model->inputs, params.wte, params.wpe, B, T,
                  C); // encoding goes into residual[0]
  for (int l = 0; l < L; l++) {

    residual = l == 0 ? acts.encoded : acts.residual3 + (l - 1) * B * T * C;

    // get the pointers of the weights for this layer
    float *l_ln1w = params.ln1w + l * C;
    float *l_ln1b = params.ln1b + l * C;
    float *l_qkvw = params.qkvw + l * 3 * C * C;
    float *l_qkvb = params.qkvb + l * 3 * C;
    float *l_attprojw = params.attprojw + l * C * C;
    float *l_attprojb = params.attprojb + l * C;
    float *l_ln2w = params.ln2w + l * C;
    float *l_ln2b = params.ln2b + l * C;
    float *l_fcw = params.fcw + l * 4 * C * C;
    float *l_fcb = params.fcb + l * 4 * C;
    float *l_fcprojw = params.fcprojw + l * C * 4 * C;
    float *l_fcprojb = params.fcprojb + l * C;

    // get the pointers of the activations for this layer
    float *l_ln1 = acts.ln1 + l * B * T * C;
    float *l_qkv = acts.qkv + l * B * T * 3 * C;
    float *l_atty = acts.atty + l * B * T * C;
    float *l_preatt = acts.preatt + l * B * NH * T * T;
    float *l_att = acts.att + l * B * NH * T * T;
    float *l_attproj = acts.attproj + l * B * T * C;
    float *l_residual2 = acts.residual2 + l * B * T * C;
    float *l_ln2 = acts.ln2 + l * B * T * C;
    float *l_fch = acts.fch + l * B * T * 4 * C;
    float *l_fch_gelu = acts.fch_gelu + l * B * T * 4 * C;
    float *l_fcproj = acts.fcproj + l * B * T * C;
    float *l_residual3 = acts.residual3 + l * B * T * C;

    // now do the forward pass
    layernorm_forward(l_ln1, residual, l_ln1w, l_ln1b, B, T, C);
    matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C);
    attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
    matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
    residual_forward(l_residual2, residual, l_attproj, B * T * C);
    layernorm_forward(l_ln2, l_residual2, l_ln2w, l_ln2b, B, T, C);
    matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4 * C);
    gelu_forward(l_fch_gelu, l_fch, B * T * 4 * C);
    matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4 * C, C);
    residual_forward(l_residual3, l_residual2, l_fcproj, B * T * C);
  }
  residual =
      acts.residual3 + (L - 1) * B * T * C; // last residual is in residual3

  layernorm_forward(acts.lnf, residual, params.lnfw, params.lnfb, B, T, C);
  matmul_forward(acts.logits, acts.lnf, params.wte, NULL, B, T, C, Vp);
  softmax_forward(acts.probs, acts.logits, B, T, V, Vp);
  cudaCheck(cudaDeviceSynchronize());
}

void gpt2_free(GPT2 *model) {
  cudaCheck(cudaFree(model->params_memory));
  cudaCheck(cudaFree(model->acts_memory));
  cudaCheck(cudaFree(model->inputs));
}

// ----------------------------------------------------------------------------
// sampler

unsigned int random_u32(uint64_t *state) {
  // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(uint64_t *state) { // random float32 in [0,1)
  return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_mult(float *probabilities, int n, float coin) {
  float cdf = 0.0f;
  for (int i = 0; i < n; i++) {
    cdf += probabilities[i];
    if (coin < cdf) {
      return i;
    }
  }
  return n - 1;
}

// ----------------------------------------------------------------------------
int main() {
  cuInit(0);
  GPT2 model;
  gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
  int B = 4;
  int T = 32; // sequence length 64 (i.e. each sequence is 64 tokens long). must
              // be <= maxT, which is 1024 for GPT-2

  Tokenizer tokenizer;
  tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

  // some memory for generating samples from the model
  uint64_t rng_state = 1337;
  int *gen_tokens = (int *)mallocCheck(B * T * sizeof(int));
  const int genT = 64; // number of steps of inference we will do
  float *cpu_logits_raw =
      (float *)mallocCheck(model.config.padded_vocab_size * sizeof(float));
  for (int i = 0; i < B * T; ++i) {
    gen_tokens[i] = tokenizer.eot_token;
  }
  printf("generating:\n---\n");
  // Time it
  cudaCheck(cudaDeviceSynchronize());
  auto start = std::chrono::high_resolution_clock::now();
  for (int t = 1; t < genT; t++) {
    gpt2_forward(&model, gen_tokens, B, T);
    float *probs = model.acts.probs + (t - 1) * model.config.padded_vocab_size;
    cudaCheck(cudaMemcpy(cpu_logits_raw, probs,
                         model.config.padded_vocab_size * sizeof(float),
                         cudaMemcpyDeviceToHost));
    float coin = random_f32(&rng_state);
    int next_token = sample_mult(cpu_logits_raw, model.config.vocab_size, coin);
    gen_tokens[t] = next_token;
  }
  cudaCheck(cudaDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  printf("Inference took %lld ms\n", (long long)duration.count());

  for (int i = 0; i < genT; ++i) {
    auto next_token = gen_tokens[i];
    if (tokenizer.init_ok) {
      const char *token_str = tokenizer_decode(&tokenizer, next_token);
      safe_printf(token_str);
    } else {
      printf("%d ", next_token);
    }
  }
  fflush(stdout);

  printf("\n---\n");

  // free
  tokenizer_free(&tokenizer);
  gpt2_free(&model);
  free(gen_tokens);
  free(cpu_logits_raw);
  return 0;
}
