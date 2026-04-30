def evaluate_mae_eV(loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            p, _ = model(b)
            preds.append(p.detach().cpu().numpy())
            trues.append(b.y.view(-1).detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    preds_eV = denorm_y(preds)
    trues_eV = denorm_y(trues)
    return float(np.mean(np.abs(preds_eV - trues_eV)))

model.eval()

y_true, y_pred = [], []
alphas = []

with torch.no_grad():
    for b in test_loader:
        b = b.to(device)
        p, a = model(b)
        y_pred.extend(p.detach().cpu().numpy().tolist())
        y_true.extend(b.y.view(-1).detach().cpu().numpy().tolist())
        alphas.append(a.detach().cpu().numpy())

y_true = denorm_y(np.array(y_true))
y_pred = denorm_y(np.array(y_pred))

mae  = float(np.mean(np.abs(y_pred - y_true)))
rmse = float(math.sqrt(np.mean((y_pred - y_true)**2)))
print("\nTEST MAE :", mae)
print("TEST RMSE:", rmse)
tol = 0.5
acc_tol = 100.0 * float(np.mean(np.abs(y_pred - y_true) <= tol))
print(f"Accuracy within ±{tol} eV:", acc_tol, "%")

rel = 1.0 - (np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-6))
acc_rel = 100.0 * float(np.mean(np.clip(rel, 0.0, 1.0)))
print("Relative Accuracy%:", acc_rel)

alphas = np.concatenate(alphas, axis=0)
print("\nMean attention weights [local, ionic, long]:", alphas.mean(axis=0))

print("\nFirst 5 predictions:")
for i in range(min(5, len(y_true))):
    print(f"{i}: true={y_true[i]:.3f}  pred={y_pred[i]:.3f}  err={abs(y_pred[i]-y_true[i]):.3f}")
