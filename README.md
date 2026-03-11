
## Master Improvement List

### PRE-PROCESSING (fix the baseline)

- [ ] Drop `street` and `country` (street has ~9000 unique values — OHE is catastrophic for linear models)
- [ ] Fix `yr_renovated`: replace `0` with `NaN` so imputer handles it correctly
- [ ] Clip `bedrooms` at 10 (there's a 33-bedroom outlier that's almost certainly a typo)
- [ ] Replace hardcoded `alpha=1.0` in Ridge with `RidgeCV` / `LassoCV` so alpha is actually tuned
- [ ] Add 5-fold CV for model selection instead of single 80/20 split

---

### FEATURE ENGINEERING

**Temporal & Age features** (from `date`, `yr_built`, `yr_renovated`)
- [ ] `house_age = sale_year - yr_built`
- [ ] `was_renovated` = binary flag (0/1)
- [ ] `effective_age` = years since renovation if renovated, else house_age
- [ ] `yrs_since_reno` = sale_year - yr_renovated (only where renovated)
- [ ] `is_peak_season` = sale_month in {3,4,5,6,7} → spring/summer commands premium
- [ ] Drop `yr_built`, `yr_renovated` after deriving these (raw years add noise)

## -------- NOTES for report ----------
BEFORE ADDING temporal features
XGBoost MAPE: 12.54%
Linear (Ridge) MAPE: 16.94%
Gap: -4.40pp

AFTER ADDING
Best linear model : Ridge
Linear Test MAPE  : 16.93%
XGBoost Test MAPE : 12.60%
Gap              : 4.32pp

Not much change - time features not helping maybe because years dont vary much? confirm with EDA before making this statement in report

**Size & Space features** (from `sqft_*`, `floors`, `bedrooms`, `bathrooms`)
- [ ] `log_sqft_living`, `log_sqft_lot` — both are right-skewed, log linearises vs price
- [ ] `sqft_per_bedroom = sqft_living / bedrooms` — spaciousness proxy
- [ ] `sqft_per_bathroom = sqft_living / bathrooms`
- [ ] `sqft_per_floor = sqft_living / floors`
- [ ] `above_basement_ratio = sqft_above / sqft_living` — layout composition
- [ ] `has_basement` = binary (sqft_basement > 0)
- [ ] `lot_utilisation = sqft_living / sqft_lot` — how "built-up" the lot is
- [ ] `total_rooms = bedrooms + bathrooms` — overall scale proxy

**Quality & Interaction features** (from `view`, `condition`, `waterfront`)
- [ ] `is_luxury` = waterfront==1 OR view>=3 (nonlinear jump in price)
- [ ] `high_view` = view >= 3 (binary)
- [ ] `good_condition` = condition >= 4 (binary)
- [ ] `size_x_condition = sqft_living * condition` — bigger + better = disproportionate premium
- [ ] `waterfront_x_sqft = waterfront * sqft_living` — waterfront premium scales with size
- [ ] `view_x_sqft = view * sqft_living`
- [ ] `bath_bed_ratio = bathrooms / bedrooms` — luxury indicator (more baths per bed)

**Location features** (from `city`, `statezip`)
- [ ] Target-encode `city` with mean(log_price) per city — fit on train only, no leakage
- [ ] Target-encode `statezip` with mean(log_price) — captures zip-level market
- [ ] Drop raw `city` / `statezip` strings after encoding (or use OHE on `city` only — ~70 values manageable)
- [ ] Frequency encode `city` as fallback (how many sales in that city = demand proxy)

**Splines & Non-linearity**
- [ ] Natural splines on `sqft_living` (5 knots) — likely curved relationship
- [ ] Natural splines on `house_age` (5 knots) — depreciation curve is non-linear
- [ ] Natural splines on `sqft_lot` (4 knots)
- [ ] Polynomial terms (degree 2) on `sqft_living`, `view`, `condition` as simpler alternative to splines
