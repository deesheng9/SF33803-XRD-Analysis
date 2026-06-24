import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths

# 1. Read directly from Excel file
file_path = "Raw Data for practical X-Ray Crystallography.xlsx"
df = pd.read_excel(file_path, skiprows=2, header=None, names=['2Theta', 'Intensity'])
df = df.dropna()

x = df['2Theta'].values
y = df['Intensity'].values

# 2. Find peaks
peaks, properties = find_peaks(y, prominence=1000, distance=10)

# 3. Sort peaks by intensity and get top 5
top_5_idx = peaks[np.argsort(y[peaks])[-5:]]
top_5_idx = np.sort(top_5_idx)

# 4. Calculate FWHM
widths_results = peak_widths(y, top_5_idx, rel_height=0.5)
widths_pts = widths_results[0]
width_heights = widths_results[1]
left_ips = widths_results[2]
right_ips = widths_results[3]

# Convert pixel width to 2Theta width
dx = x[1] - x[0]
fwhm_2theta = widths_pts * dx

# 5. Scherrer equation calculations
K = 0.9
lambda_nm = 0.154 

results = []
for i in range(len(top_5_idx)):
    peak_idx = top_5_idx[i]
    theta_deg = x[peak_idx] / 2
    beta_rad = np.deg2rad(fwhm_2theta[i])
    theta_rad = np.deg2rad(theta_deg)
    
    D_nm = (K * lambda_nm) / (beta_rad * np.cos(theta_rad))
    
    results.append({
        "Peak": i + 1,
        "2Theta": x[peak_idx],
        "Intensity": y[peak_idx],
        "Beta_rad": beta_rad,
        "Theta_deg": theta_deg,
        "D_nm": D_nm
    })

# Print results
print("-" * 75)
print(f"{'Peak':<6} | {'2Theta (°)':<10} | {'Intensity':<10} | {'Beta (rad)':<10} | {'Theta (°)':<10} | {'D (nm)':<8}")
print("-" * 75)
for r in results:
    print(f"{r['Peak']:<6} | {r['2Theta']:<10.3f} | {r['Intensity']:<10.1f} | {r['Beta_rad']:<10.5f} | {r['Theta_deg']:<10.3f} | {r['D_nm']:<8.2f}")
print("-" * 75)

# 6. Plotting
fig = plt.figure(figsize=(15, 10))

# Main plot
ax_main = plt.subplot(2, 1, 1)
ax_main.plot(x, y, color='#2c3e50')
ax_main.plot(x[top_5_idx], y[top_5_idx], "x", color='red', markersize=10)
ax_main.set_title("Full XRD Pattern", fontsize=14, fontweight='bold')
ax_main.set_xlabel("2Theta (degrees)")
ax_main.set_ylabel("Intensity (cps)")
ax_main.grid(True, linestyle='--', alpha=0.6)

# Subplots for 5 peaks
for i in range(len(top_5_idx)):
    ax = plt.subplot(2, 5, i + 6)
    peak_x = x[top_5_idx[i]]
    mask = (x > peak_x - 1.5) & (x < peak_x + 1.5)
    
    ax.plot(x[mask], y[mask], color='#2980b9')
    ax.hlines(y=width_heights[i], xmin=x[0]+left_ips[i]*dx, xmax=x[0]+right_ips[i]*dx, color='red', linewidth=2)
    ax.set_title(f"Peak {i+1}")
    ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("XRD_Analysis_Result.png", dpi=300)
plt.show()