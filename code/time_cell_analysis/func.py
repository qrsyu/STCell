import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm

def _timecell_model(t, a0, a1, mu, sigma):
    return a0 + a1 * np.exp(-0.5 * ((t - mu)/sigma)**2)

def _fit_model(rate, t, mu_bounds, sigma_bounds):

    # Fit the time cell model 1
    p_init1 = [rate.mean(), (rate.max()-rate.min()), t[np.argmax(rate)], (t[-1]-t[0])/10]
    bounds1 = ([0, 0,          mu_bounds[0], sigma_bounds[0]],
              [1, rate.max(), mu_bounds[1], sigma_bounds[1]])
    popt1, _ = curve_fit(_timecell_model, t, rate, p0=p_init1, bounds=bounds1, maxfev=5000)
    
    # Compute the error sums of squares
    y1 = _timecell_model(t, *popt1)
    sse1 = np.sum((rate - y1)**2)

    return sse1, popt1

def classify_timecell(rates, T_total, threshold=2, nperm=500):
    n_neurons, n_bins = rates.shape
    t = np.linspace(0, T_total, n_bins, endpoint=False)
    mu_bounds = (0.0, T_total)
    sigma_bounds = (1e-3, T_total)

    results = []
    for i in tqdm(range(n_neurons)):
        rate = rates[i]

        # Calculate observed ΔR^2
        Err, popt = _fit_model(rate, t, mu_bounds, sigma_bounds)

        is_time = Err < threshold
        
        results.append({
            "neuron":       i,
            "is_time_cell": bool(is_time),
            'Err':    float(Err),
            'popt':  popt,
            })
    return results

