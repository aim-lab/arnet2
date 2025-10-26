#!/usr/bin/env python3
"""
Export ArNet2 model as a TensorFlow SavedModel for secure inference.

Inputs:
  - x            float32 [N, 60]   (RR features)
  - prec_windows float32 [N]       (preceding windows count)
  - glob_lab     float32 [N]       (ignored by TF path; kept for API parity)
  - ids          string  [N]       (sample/patient ids)

Outputs:
  - probs float32 [N]   (AF probability per sample)
  - pred  int32   [N]   (binary classification: 0/1)
"""

import argparse
import tensorflow as tf
from models.model_utils import load_model


def parse_args():
    p = argparse.ArgumentParser(description="Export ArNet2 model to TensorFlow SavedModel")
    p.add_argument("--model", required=True, help="Path to ArNet2 .pkl model file")
    p.add_argument("--feature_extractor", required=True, help="Path to ResNet/1D-CNN .pkl file")
    p.add_argument("--output", default="exported_model", help="Output SavedModel directory")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Default fixed threshold for 'predict_fixed' (default: 0.5)")
    return p.parse_args()


class ArNet2Wrapper(tf.Module):
    """
    Trackable wrapper that:
      • Tracks all TF variables used by ArNet2 (feature extractor + GRU heads)
      • Reassembles inputs to the mixed-string matrix expected by arnet2.predict_proba_tf
      • Calls arnet2.predict_proba_tf (pure TF)
    """

    def __init__(self, arnet2, default_threshold: float):
        super().__init__()
        self.arnet2 = arnet2
        self.default_threshold = float(default_threshold)

        # ---- Track TF objects so SavedModel knows all variables belong to this wrapper ----
        # Feature extractor Keras model
        self.fe = arnet2.feature_extractor.model
        # All GRU heads (in deterministic label order)
        self.labels_py = list(arnet2.labels.tolist())
        self.heads = [arnet2.models[lab] for lab in self.labels_py]

    @staticmethod
    def _assemble_string_matrix(x, prec_windows, glob_lab, ids):
        """
        Build the mixed string matrix [N,63]:
          0..59 -> rr (strings)
          60    -> prec_windows (string)
          61    -> glob_lab (string)  # not used by TF path, kept for API parity
          62    -> ids (string)
        """
        x = tf.convert_to_tensor(x, tf.float32)              # [N,60]
        prec_windows = tf.convert_to_tensor(prec_windows)    # [N]
        glob_lab = tf.convert_to_tensor(glob_lab)            # [N]
        ids = tf.convert_to_tensor(ids, tf.string)           # [N]

        rr_s = tf.as_string(x)                               # [N,60] string
        pw_s = tf.as_string(prec_windows)  # already int32 -> correct integer string
        gl_s = tf.as_string(tf.cast(glob_lab, tf.float32))      # [N]

        pw_s = tf.expand_dims(pw_s, axis=1)                  # [N,1]
        gl_s = tf.expand_dims(gl_s, axis=1)                  # [N,1]
        ids  = tf.expand_dims(ids,  axis=1)                  # [N,1]

        return tf.concat([rr_s, pw_s, gl_s, ids], axis=1)    # [N,63] string

    @tf.function(
        input_signature=[
            tf.TensorSpec([None, 60], tf.float32, name="x"),
            tf.TensorSpec([None], tf.int32, name="prec_windows"),
            tf.TensorSpec([None], tf.float32, name="glob_lab"),
            tf.TensorSpec([None], tf.string,  name="ids"),
            tf.TensorSpec([], tf.float32,     name="threshold"),
        ]
    )
    def call_with_threshold(self, x, prec_windows, glob_lab, ids, threshold):
        # Reassemble to the format ArNet2.predict_proba_tf expects
        X_str = self._assemble_string_matrix(x, prec_windows, glob_lab, ids)  # [N,63] string

        # Pure-TF inference from your class; returns [N,2] ([:,1] is P(AF))
        probs_2c = self.arnet2.predict_proba_tf(X_str)  # [N,2] float32
        probs = probs_2c[:, 1]                          # [N]
        pred = tf.cast(probs > threshold, tf.int32)     # [N] 0/1
        return {"probs": probs, "pred": pred}

    @tf.function(
        input_signature=[
            tf.TensorSpec([None, 60], tf.float32, name="x"),
            tf.TensorSpec([None], tf.int32, name="prec_windows"),  # <-- change here
            tf.TensorSpec([None], tf.float32, name="glob_lab"),  # stays float
            tf.TensorSpec([None], tf.string, name="ids"),
        ]
    )
    def predict_fixed(self, x, prec_windows, glob_lab, ids):
        return self.call_with_threshold(
            x, prec_windows, glob_lab, ids,
            tf.constant(self.default_threshold, dtype=tf.float32),
        )

    @tf.function(
        input_signature=[
            tf.TensorSpec([None, 60], tf.float32, name="x"),
            tf.TensorSpec([], tf.float32,     name="threshold"),

        ]
    )
    def predict_windows(self, x, threshold):
        if hasattr(self.arnet2.feature_extractor, "predict_proba_tf"):
            probs2 = self.arnet2.feature_extractor.predict_proba_tf(x)  # [N,2]
            probs  = probs2[:, 1]
        else:
            # Fallback: use FE Keras model directly
            logits = self.arnet2.feature_extractor.model(x, training=False)  # [N,1] or [N,2]
            logits = tf.convert_to_tensor(logits, tf.float32)
            if logits.shape.rank == 2 and logits.shape[-1] == 2:
                probs = tf.nn.softmax(logits, axis=-1)[:, 1]
            else:
                probs = tf.nn.sigmoid(tf.squeeze(logits, axis=-1))
        pred = tf.cast(probs > threshold, tf.int32)     # [N] 0/1
        return {"probs": probs, "pred": pred}

def main():
    args = parse_args()

    print("\n===== ArNet2 SavedModel Export =====")
    print("Loading model from:", args.model)
    print("Loading feature extractor from:", args.feature_extractor)

    model_dict = load_model(
        path=args.model,
        algo="ArNet2",
        path_feature_extractor=args.feature_extractor,
    )
    arnet2 = model_dict["classifier"]
    default_th = float(model_dict.get("best_th", args.threshold))
    print(f"Default threshold (predict_fixed): {default_th}")

    print("Wrapping & tracking variables…")
    wrapper = ArNet2Wrapper(arnet2, default_th)

    print("Exporting to:", args.output)
    signatures = {
        "serving_default": wrapper.call_with_threshold.get_concrete_function(
            tf.TensorSpec([None, 60], tf.float32, name="x"),
            tf.TensorSpec([None], tf.int32, name="prec_windows"),
            tf.TensorSpec([None], tf.float32, name="glob_lab"),
            tf.TensorSpec([None], tf.string, name="ids"),
            tf.TensorSpec([], tf.float32, name="threshold"),
        ),
        "predict_fixed": wrapper.predict_fixed.get_concrete_function(
            tf.TensorSpec([None, 60], tf.float32, name="x"),
            tf.TensorSpec([None], tf.int32, name="prec_windows"),
            tf.TensorSpec([None], tf.float32, name="glob_lab"),
            tf.TensorSpec([None], tf.string, name="ids"),
        ),
        "predict_windows": wrapper.predict_windows.get_concrete_function(
            tf.TensorSpec([None, 60], tf.float32, name="x"),
            tf.TensorSpec([], tf.float32, name="threshold"),
        ),
    }

    tf.saved_model.save(wrapper, args.output, signatures=signatures)

    print("Export complete!")
    print(f"SavedModel folder: {args.output}")
    print("Signatures available: [serving_default, predict_fixed]")
    print("====================================\n")


if __name__ == "__main__":
    main()
