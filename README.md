# Spatial-Microbiome-Digital-Twins

## Data Collection

Our study collected four public spatially structured microbiome datasets for benchmarking. These datasets span centimeter- to kilometer-scale gradients.

**Body sites dataset**: This human cohort comprises 1,628 samples from four anatomical niches—tongue, left palm, right palm and faeces—capturing centimetre-scale separation within hosts. Microbial community composition was measured via 16S ribosomal RNA (rRNA) amplicon sequencing. For the taxonomic resolution analysis, we used the provided rank-specific abundance tables (from class to genus).

**HUST campus dataset**: This environmental dataset contains 102 environmental microbiomes collected on the Huazhong University of Science and Technology (HUST) campus across 11 functional locations. The locations—Dormitory, Library, Station, Bus, Field, Sports, Gate, Clinic, Hotel, Lake, Hill—are ordered from higher to lower human traffic. Location labels and timestamps were taken directly from the source metadata. To capture repeated surveys, we defined subject id by sampling time. Microbial community composition was measured via 16S rRNA amplicon sequencing.

**Ocean depth dataset**: This dataset profiles 148 samples across five depth layers (0–4,500 m), representing a strong vertical physicochemical gradient. Samples originated from two expeditions, Malaspina-2010 and Hotmix. Depth layer labels were harmonised to a five-level factor for downstream evaluation.

**Ocean currents dataset**: The Ocean currents dataset was derived from the global surface-ocean survey reported in "Structure and function of the global ocean microbiome". That study released taxonomic profiles for hundreds of stations sampled during the Tara Oceans expedition, but did not provide an explicit “current-segment” annotation. To create a spatially coherent gradient for benchmarking digital-twin imputation, we mapped stations to a single trans-basin current system using their reported coordinates. A subset of 68 surface-water communities were filtered for modeling.

## Benchmark Methods

We implemented five representative benchmark models to evaluate their predictive performance on spatially structured microbiome data: **Random Forest**, **ElasticNet**, **LSTM**, **Lasso**, and **PatchTST**. The models span linear, ensemble, and deep learning architectures, as well as domain-specific strategies for microbiome data. Deep models were adapted to spatially ordered inputs by indexing samples along the relevant gradient (anatomical site, campus location, depth layer or current segment). For each dataset, the data were split at the community level into training (70%) and testing sets (30%), repeated 10 times to ensure robustness.

- For the **body sites dataset**, the model received the left-palm, right-palm, and oral community profiles as inputs and predicts the gut profile from the same instance.
- For the other cohorts, the model was given the first 30% of positions as context and is required to predict all subsequent positions along that gradient.

### Random Forest

An ensemble learning method that constructs multiple decision trees and aggregates their outputs. Implemented using `RandomForestRegressor` (scikit-learn) with 100 trees (`n_estimators=100`) and `max_depth=10`. Input features were standardized, and training minimized MSE. Microbial feature importances were averaged across time.

### Elastic Net

A regularized linear regression combining L1 and L2 penalties. Implemented with `ElasticNet` (scikit-learn), using `alpha=0.1`, `l1_ratio=0.5`, and standardized inputs. The regularized loss is:

```
L(w) = (1/2n) ∑(yᵢ - xᵢᵀw)² + α(λ‖w‖₁ + (1-λ)/2 ‖w‖₂²)
```

### Long Short-Term Memory (LSTM)

A recurrent neural network (RNN) model capturing long-range dependencies. Implemented with PyTorch as a sequence-to-sequence LSTM with ReLU output, trained with Adam optimizer (`lr=0.001`, hidden dim=32, 100 epochs). Custom dataloader and autoregressive prediction used.

### Lasso Regression

A sparse linear model using L1 regularization. Implemented with `Lasso` (scikit-learn), `alpha=0.1`. Z-score normalized inputs and fixed random seed (`random_state=0`) ensure reproducibility.

### PatchTST

A Transformer-based time-series model for microbiome data under irregular sampling. Core model uses a temporal Transformer encoder-decoder with self-attention:

```
Attention(Q,K,V) = Softmax((QKᵀ)/√dₖ)V
```

Trained with `lr=1e-4`, batch size 64, up to 8000 epochs with early stopping. The loss function is:

```
Loss = Eⱼ ∑ ‖x̂ᵢ(t+Δt) - xᵢ(t+Δt)‖²
```

Leave-one-out cross-validation was used across fermentation workshops.

---
