from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_unflatten

from mflux.weights.download import snapshot_download
from mflux.weights.lora_converter import LoRAConverter
from mflux.weights.weight_util import WeightUtil


@dataclass
class MetaData:
    quantization_level: int | None = None
    scale: float | None = None
    is_lora: bool = False
    mflux_version: str | None = None


class WeightHandler:
    def __init__(
        self,
        meta_data: MetaData,
        clip_encoder: dict | None = None,
        t5_encoder: dict | None = None,
        vae: dict | None = None,
        transformer: dict | None = None,
    ):
        self.clip_encoder = clip_encoder
        self.t5_encoder = t5_encoder
        self.vae = vae
        self.transformer = transformer
        self.meta_data = meta_data

    @staticmethod 
    def load_custom_encoder_weights(
        encoder_path: str,
        encoder_type: str  # "clip" or "t5"
    ) -> dict:
        """Load weights for a custom encoder from HuggingFace cache (NO DOWNLOADING)"""
        # Find the model in HuggingFace cache without downloading
        custom_encoder_path = WeightHandler._find_cached_model(encoder_path)
        
        if encoder_type == "clip":
            # For CLIP encoders, load from text_encoder directory
            encoder_weights, _, _ = WeightHandler._load_clip_encoder(custom_encoder_path)
        elif encoder_type == "t5":
            # For standalone T5 encoders, load directly from model root
            encoder_weights, _, _ = WeightHandler._load_standalone_t5_encoder(custom_encoder_path)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
            
        return encoder_weights

    @staticmethod
    def _find_cached_model(repo_id: str) -> Path:
        """Find model in HuggingFace cache without downloading"""
        try:
            from huggingface_hub import scan_cache_dir
        except ImportError:
            raise ImportError("huggingface_hub required to access cache")
        
        print(f"[mflux] DEBUG: Looking for {repo_id} in HuggingFace cache")
        
        try:
            cache_info = scan_cache_dir()
            
            for repo in cache_info.repos:
                if repo.repo_id == repo_id:
                    print(f"[mflux] DEBUG: Found {repo_id} in cache")
                    if len(repo.revisions) > 0:
                        # Get the latest revision's snapshot path
                        latest_revision = next(iter(repo.revisions))
                        snapshot_path = Path(latest_revision.snapshot_path)
                        print(f"[mflux] DEBUG: Using cached path: {snapshot_path}")
                        return snapshot_path
                    else:
                        raise FileNotFoundError(f"No revisions found for {repo_id} in cache")
            
            raise FileNotFoundError(f"Model {repo_id} not found in HuggingFace cache. Please download it first.")
            
        except Exception as e:
            print(f"[mflux] DEBUG: Error scanning cache: {e}")
            raise

    @staticmethod
    def _load_standalone_t5_encoder(root_path: Path) -> tuple[dict, int, str | None]:
        """Load weights for a standalone T5 encoder model"""
        import mlx.core as mx
        
        print(f"[mflux] DEBUG: Loading T5 from path: {root_path}")
        print(f"[mflux] DEBUG: Path exists: {root_path.exists()}")
        
        if not root_path.exists():
            raise FileNotFoundError(f"T5 encoder path does not exist: {root_path}")
        
        # List all files for debugging
        all_files = list(root_path.iterdir())
        print(f"[mflux] DEBUG: Files in T5 directory: {[f.name for f in all_files[:10]]}")  # Show first 10
        
        weights = []
        quantization_level = None
        mflux_version = None

        # Look for .safetensors files directly in root
        safetensors_files = list(root_path.glob("*.safetensors"))
        print(f"[mflux] DEBUG: Found {len(safetensors_files)} safetensors files")
        
        if not safetensors_files:
            # Fallback to model.safetensors
            safetensors_files = list(root_path.glob("model.safetensors"))
            if not safetensors_files:
                raise FileNotFoundError(f"No safetensors files found in {root_path}")

        for file in sorted(safetensors_files):
            print(f"[mflux] DEBUG: Loading weights from: {file.name}")
            data = mx.load(str(file), return_metadata=True)
            weight = list(data[0].items())
            if len(data) > 1:
                quantization_level = data[1].get("quantization_level")
                mflux_version = data[1].get("mflux_version")
            weights.extend(weight)

        # Convert to dict format
        weights_dict = dict(weights)
        print(f"[mflux] DEBUG: Loaded {len(weights_dict)} weight tensors")
        print(f"[mflux] DEBUG: Top-level keys: {list(weights_dict.keys())[:10]}")  # Show first 10 keys

        # Process T5 weights to match expected format (same as _load_t5_encoder)
        if quantization_level is not None:
            print(f"[mflux] DEBUG: Using pre-quantized weights (level: {quantization_level})")
            return weights_dict, quantization_level, mflux_version

        # Reshape and process the huggingface weights for standalone T5
        if "encoder" in weights_dict:
            print(f"[mflux] DEBUG: Processing HuggingFace T5 format")
            weights_dict["final_layer_norm"] = weights_dict["encoder"]["final_layer_norm"]
            
            blocks = weights_dict["encoder"]["block"]
            print(f"[mflux] DEBUG: Processing {len(blocks)} T5 blocks")
            
            for i, block in enumerate(blocks):
                try:
                    if "layer" not in block:
                        print(f"[mflux] DEBUG: Block {i} missing 'layer' key, skipping")
                        continue
                    
                    layers = block["layer"]
                    if len(layers) < 2:
                        print(f"[mflux] DEBUG: Block {i} has only {len(layers)} layers, expected 2")
                        continue
                        
                    attention = layers[0]
                    ff = layers[1]
                    block.pop("layer")
                    block["attention"] = attention
                    block["ff"] = ff
                except (KeyError, IndexError) as e:
                    print(f"[mflux] DEBUG: Error processing block {i}: {e}")
                    continue

            weights_dict["t5_blocks"] = weights_dict["encoder"]["block"]

            # Handle relative_attention_bias
            if weights_dict["t5_blocks"]:
                first_block = weights_dict["t5_blocks"][0]
                if ("attention" in first_block and 
                    "SelfAttention" in first_block["attention"] and
                    "relative_attention_bias" in first_block["attention"]["SelfAttention"]):
                    
                    relative_attention_bias = first_block["attention"]["SelfAttention"]["relative_attention_bias"]
                    print(f"[mflux] DEBUG: Copying relative_attention_bias to {len(weights_dict['t5_blocks'])-1} other blocks")
                    
                    for block in weights_dict["t5_blocks"][1:]:
                        if ("attention" in block and 
                            "SelfAttention" in block["attention"]):
                            block["attention"]["SelfAttention"]["relative_attention_bias"] = relative_attention_bias

            weights_dict.pop("encoder")
            print(f"[mflux] DEBUG: Successfully processed standalone T5 encoder")
        else:
            print(f"[mflux] DEBUG: No 'encoder' key found, using weights as-is")

        return weights_dict, quantization_level, mflux_version

    @staticmethod
    def load_regular_weights(
        repo_id: str | None = None,
        local_path: str | None = None,
        transformer_repo_id: str | None = None,
    ) -> "WeightHandler":
        # Load the weights from disk, huggingface cache, or download from huggingface
        root_path = Path(local_path) if local_path else WeightHandler._download_or_get_cached_weights(repo_id)

        # Some custom models might have a different specific transformer setup
        if transformer_repo_id:
            transformer_path = WeightHandler._download_transformer_weights(transformer_repo_id)
        else:
            transformer_path = root_path

        # Load the weights
        transformer, quantization_level, mflux_version = WeightHandler.load_transformer(root_path=transformer_path)
        clip_encoder, _, _ = WeightHandler._load_clip_encoder(root_path=root_path)
        t5_encoder, _, _ = WeightHandler._load_t5_encoder(root_path=root_path)
        vae, _, _ = WeightHandler._load_vae(root_path=root_path)

        return WeightHandler(
            clip_encoder=clip_encoder,
            t5_encoder=t5_encoder,
            vae=vae,
            transformer=transformer,
            meta_data=MetaData(
                quantization_level=quantization_level,
                scale=None,
                is_lora=False,
                mflux_version=mflux_version,
            ),
        )

    def num_transformer_blocks(self) -> int:
        return len(self.transformer["transformer_blocks"])

    def num_single_transformer_blocks(self) -> int:
        return len(self.transformer["single_transformer_blocks"])

    @staticmethod
    def _load_clip_encoder(root_path: Path) -> tuple[dict, int, str | None]:
        weights, quantization_level, mflux_version = WeightHandler.get_weights("text_encoder", root_path)
        return weights, quantization_level, mflux_version

    @staticmethod
    def _load_t5_encoder(root_path: Path) -> tuple[dict, int, str | None]:
        weights, quantization_level, mflux_version = WeightHandler.get_weights("text_encoder_2", root_path)

        # Quantized weights (i.e. ones exported from this project) don't need any post-processing.
        if quantization_level is not None:
            return weights, quantization_level, mflux_version

        # Reshape and process the huggingface weights
        weights["final_layer_norm"] = weights["encoder"]["final_layer_norm"]
        for block in weights["encoder"]["block"]:
            attention = block["layer"][0]
            ff = block["layer"][1]
            block.pop("layer")
            block["attention"] = attention
            block["ff"] = ff

        weights["t5_blocks"] = weights["encoder"]["block"]

        # Only the first layer has the weights for "relative_attention_bias", we duplicate them here to keep code simple
        relative_attention_bias = weights["t5_blocks"][0]["attention"]["SelfAttention"]["relative_attention_bias"]
        for block in weights["t5_blocks"][1:]:
            block["attention"]["SelfAttention"]["relative_attention_bias"] = relative_attention_bias

        weights.pop("encoder")
        return weights, quantization_level, mflux_version

    @staticmethod
    def _safely_extract_ff_weights(ff_dict: dict) -> dict | None:
        try:
            net = ff_dict["net"]
            return {
                "linear1": net[0]["proj"],
                "linear2": net[2],
            }
        except (KeyError, IndexError, TypeError):
            return None

    @staticmethod
    def load_transformer(root_path: Path | None = None, lora_path: str | None = None) -> tuple[dict, int, str | None]:
        weights, quantization_level, mflux_version = WeightHandler.get_weights("transformer", root_path, lora_path)

        if lora_path:
            if "transformer" not in weights:
                weights = LoRAConverter.load_weights(lora_path)
            weights = weights["transformer"]

        # Quantized weights (i.e. ones exported from this project) don't need any post-processing.
        if quantization_level or mflux_version:
            return weights, quantization_level, mflux_version

        # Reshape and process the huggingface weights
        if "transformer_blocks" in weights:
            for block in weights["transformer_blocks"]:
                # Safely process ff weights
                if "ff" in block:
                    extracted_ff = WeightHandler._safely_extract_ff_weights(block["ff"])
                    if extracted_ff is not None:
                        block["ff"] = extracted_ff

                # Safely process ff_context weights
                if "ff_context" in block:
                    extracted_ff_context = WeightHandler._safely_extract_ff_weights(block["ff_context"])
                    if extracted_ff_context is not None:
                        block["ff_context"] = extracted_ff_context

        return weights, quantization_level, mflux_version

    @staticmethod
    def _load_vae(root_path: Path) -> tuple[dict, int, str | None]:
        weights, quantization_level, mflux_version = WeightHandler.get_weights("vae", root_path)

        # Quantized weights (i.e. ones exported from this project) don't need any post-processing.
        if quantization_level is not None:
            return weights, quantization_level, mflux_version

        # Reshape and process the huggingface weights
        weights["decoder"]["conv_in"] = {"conv2d": weights["decoder"]["conv_in"]}
        weights["decoder"]["conv_out"] = {"conv2d": weights["decoder"]["conv_out"]}
        weights["decoder"]["conv_norm_out"] = {"norm": weights["decoder"]["conv_norm_out"]}
        weights["encoder"]["conv_in"] = {"conv2d": weights["encoder"]["conv_in"]}
        weights["encoder"]["conv_out"] = {"conv2d": weights["encoder"]["conv_out"]}
        weights["encoder"]["conv_norm_out"] = {"norm": weights["encoder"]["conv_norm_out"]}
        return weights, quantization_level, mflux_version

    @staticmethod
    def _get_model_file_pattern(model_name: str, root_path: Path):
        if model_name == "transformer":
            nested_files = list(root_path.glob("transformer/*.safetensors"))
            if nested_files:
                return root_path.glob("transformer/*.safetensors")
            else:
                return root_path.glob("*.safetensors")
        else:
            return root_path.glob(model_name + "/*.safetensors")

    @staticmethod
    def get_weights(
        model_name: str,
        root_path: Path | None = None,
        lora_path: str | None = None,
    ) -> tuple[dict, int, str | None]:
        weights = []
        quantization_level = None
        mflux_version = None

        if root_path is not None:
            file_glob = WeightHandler._get_model_file_pattern(model_name, root_path)
            for file in sorted(file_glob):
                data = mx.load(str(file), return_metadata=True)
                weight = list(data[0].items())
                if len(data) > 1:
                    quantization_level = data[1].get("quantization_level")
                    mflux_version = data[1].get("mflux_version")
                weights.extend(weight)

        if lora_path and root_path is None:
            data = mx.load(lora_path, return_metadata=True)
            weight = list(data[0].items())
            if len(data) > 1:
                mflux_version = data[1].get("mflux_version", None)
            weights.extend(weight)

        # Non huggingface weights (i.e. ones exported from this project) don't need any reshaping.
        if quantization_level is not None or mflux_version is not None:
            return tree_unflatten(weights), quantization_level, mflux_version

        # Huggingface weights needs to be reshaped
        weights = [WeightUtil.reshape_weights(k, v) for k, v in weights]
        weights = WeightUtil.flatten(weights)
        unflatten = tree_unflatten(weights)
        return unflatten, quantization_level, mflux_version

    @staticmethod
    def _download_or_get_cached_weights(repo_id: str) -> Path:
        return Path(
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=[
                    "text_encoder/*.safetensors",
                    "text_encoder_2/*.safetensors",
                    "transformer/*.safetensors",
                    "vae/*.safetensors",
                ],
            )
        )

    @staticmethod
    def _download_transformer_weights(repo_id: str) -> Path:
        return Path(
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=[
                    "transformer/*.safetensors",
                    "*.safetensors",
                ],
            )
        )
