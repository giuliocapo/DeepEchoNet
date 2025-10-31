import torch
import torch.nn as nn
from torchinfo import summary
import io

def summarize_model(model: nn.Module, input_size=(1, 3, 64, 64), arch_name="unknown", save_path=None):
    """
    Genera un riepilogo Markdown leggibile dell'architettura corrente.
    Usa torchinfo.summary (simile a keras.summary()) e produce una versione compatta.
    """
    try:
        from torchinfo import summary
    except ImportError:
        raise ImportError("⚠️ Installa torchinfo con: pip install torchinfo")

    buffer = io.StringIO()
    print(f"\n🧠 Current Architecture: {arch_name}", file=buffer)
    print("─" * 60, file=buffer)
    print(f"Input size: {input_size}", file=buffer)
    print("─" * 60, file=buffer)

    # crea il summary dettagliato
    model_summary = summary(model, input_size=input_size, verbose=0, col_names=("input_size", "output_size", "num_params"))
    print(model_summary, file=buffer)

    total_params = sum(p.numel() for p in model.parameters())
    print("─" * 60, file=buffer)
    print(f"Total parameters: {total_params/1e6:.2f}M", file=buffer)

    md = buffer.getvalue()

    # opzionale: salva su file
    if save_path:
        with open(save_path, "w") as f:
            f.write("```text\n" + md + "\n```")
        print(f"[INFO] Architecture summary saved → {save_path}")

    return md
