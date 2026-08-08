"""The three square classifiers. Input is one 144x144x4 square crop (RGB + the square's mask)
plus a small metadata vector; output is a piece label for that square.

Model 1 is the original baseline, model 2 adds normalisation/residuals/dilation, and model 3
(SquareClassifierMultiHead) is the one actually shipped -- see its docstring for why the head is
factored into three.
"""

import torch
from torch import nn


class SquareClassifier(nn.Module):
    """Model 1, the baseline: a plain conv stack (no normalisation, no residuals), max-pooled
    down after every layer, then global max+avg pooling concatenated with the metadata vector and
    fed to an MLP with a single 13-way head.

    The metadata exists because the same piece looks completely different depending on where the
    camera sits: which board corner is at the top of the image decides how a piece should be
    interpreted, and the robot's height changes the apparent shape on top of that. Both are
    captured well enough by a one-hot of the top-left corner, so that is what gets fed in. The
    mask channel plays the same role for "where in the crop should I even look".

    Kept for reference; superseded by SquareClassifier2 and SquareClassifierMultiHead. Its size is
    the reason those exist: ~450k parameters against only ~10k square crops of training data, which
    is what pushed the project towards data augmentation and smaller models.
    """
    def __init__(self):
        super().__init__()
        # Downsample using max pool
        self.max_pool_1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Global max pool at end
        self.max_pool_2 = nn.MaxPool2d(kernel_size=16)
        # Global average pool at end
        self.avg_pool_1 = nn.AvgPool2d(kernel_size=16)

        # ReLU
        self.relu = nn.ReLU()

        ### Image feature extraction
        self.image_feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, padding=1),
            self.relu,
            self.max_pool_1,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            self.relu, 
            self.max_pool_1, 
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            self.relu,
            self.max_pool_1,
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            self.relu
        )        

        ### Final MLP
        self.mlp_2 = nn.Sequential(
            nn.Linear(512 + 4, 256),
            self.relu,
            nn.Linear(256, 128),
            self.relu,
            nn.Linear(128, 13)
        )
    
    def forward(self, image, metadata):
        image_features = self.image_feature_extraction(image)
        image_features = torch.cat(
            [
                torch.flatten(nn.MaxPool2d(16)(image_features), start_dim=1), # want to maintain batch structure
                torch.flatten(nn.AvgPool2d(16)(image_features), start_dim=1)
            ],
            dim=1
        )
        mlp_input = torch.cat([image_features, metadata], dim=1)
        logits = self.mlp_2(mlp_input)
        return logits


class SquareClassifier2(nn.Module):
    """Model 2: convolutional feature extraction, then the features are concatenated with the
    metadata and passed through fully connected layers to a single 13-way head.

    Over the baseline this adds batch normalisation, a residual block, depthwise dilated
    convolutions to widen the receptive field, and 2x2 adaptive max/avg pooling that preserves
    some spatial structure instead of collapsing it entirely.
    """
    def __init__(self):
        super().__init__()
        # Downsample using max pool
        self.max_pool_1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # ReLU
        self.relu = nn.ReLU()

        ### Image feature extraction
        self.image_feature_extraction_1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            self.relu,
            self.max_pool_1,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            self.relu, 
            self.max_pool_1, 
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            self.relu,
            self.max_pool_1
        )        
        self.image_feature_extraction_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            self.relu,
            # Increase receptive field of neurons in final layer through a series
            # of depthwise dilated convolutions, followed by a channel mix-in
            nn.Conv2d(128, 128, kernel_size=3, groups=128, padding=1),
            nn.BatchNorm2d(128),
            self.relu,
            nn.Conv2d(128, 128, kernel_size=3, groups=128, dilation=2, padding=2, bias=False),
            nn.BatchNorm2d(128),
            self.relu,
            nn.Conv2d(128, 128, kernel_size=3, groups=128, dilation=5, padding=5, bias=False),
            nn.BatchNorm2d(128),
            self.relu,
            # 1x1 Channel mix-in
            nn.Conv2d(128, 128, kernel_size=1, groups=1, dilation=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
        )
        # Initialise the weights (scale) of the last batch norm to zero; so by default
        # image_feature_extraction_2 is the identiy
        nn.init.zeros_(self.image_feature_extraction_2[-1].weight)


        # Adaptive Average Pooling at end; 2x2 to preserve SOME spatial structure
        # Improvements to this would perhaps take the mask into account more explicitly
        self.max_pool = nn.AdaptiveMaxPool2d((2, 2))
        self.avg_pool = nn.AdaptiveAvgPool2d((2, 2))

        ### Final MLP
        self.mlp_2 = nn.Sequential(
            nn.Linear(1024+4, 128),
            self.relu,
            nn.Linear(128, 64),
            self.relu,
            nn.Linear(64, 13)
        )
    
    def forward(self, image, metadata):
        image_features = self.image_feature_extraction_1(image)
        # Add a residual connection for this part, which is quite deep
        image_features = self.relu(image_features + self.image_feature_extraction_2(image_features))
        image_features = torch.cat(
            (
                torch.flatten(self.max_pool(image_features), start_dim=1),
                torch.flatten(self.avg_pool(image_features), start_dim=1)
            ),
            dim=1
        )

        # Batch norm assumes iid batches, and metadata varies only across setups: a batch drawn
        # from very few setups would see near-zero variance in the metadata features. With
        # batch_size=64 and shuffle=True that is very unlikely, and the failure mode is mild
        # anyway (a constant feature just carries no information). Moot in any case now that the
        # metadata is only a one-hot of the top-left corner.
        mlp_input = torch.cat([image_features, metadata], dim=1)
        logits = self.mlp_2(mlp_input)
        return logits


class SquareClassifierMultiHead(nn.Module):
    """Model 3, the one that ships. Same trunk as SquareClassifier2 (conv feature extraction,
    residual dilated block, adaptive max/avg pooling, image features concatenated with metadata
    -> mlp_input), but the single 13-way head is replaced by three:
      - empty_head: 1 logit (empty vs non-empty)
      - color_head: 2 logits (white/black | non-empty)
      - type_head:  6 logits (K/Q/R/B/N/P | non-empty)

    A single 13-way head has to learn each (colour, type) combination from only the crops of that
    exact piece, and the 13 classes are badly imbalanced: ~56% of squares are empty, and there is
    exactly one king and one queen per side against eight pawns. Factoring the problem lets every
    head pool data across the other factor -- the colour head sees every piece regardless of type,
    the type head sees every king regardless of colour -- so the data-starved king/queen classes
    get twice as many training data points and the huge empty class is handled by its own
    binary head instead of dominating a 13-way softmax.

    The heads are recombined into the original 13-way distribution at inference/eval time via
    reconstruct_13way_logprobs.

    `log_prior` is the 13-way training-set log-prior used for optional Bayesian prior correction
    (see reconstruct_13way_logprobs). It lives on the model as a buffer rather than being passed
    around separately so that it travels with the weights: buffers are non-trainable, land in
    state_dict, and follow .to(device). It is registered UNCONDITIONALLY (all-zeros when unknown,
    which subtracts nothing) so that state_dict keys never depend on how the model was
    constructed - registering it only sometimes makes checkpoints fail to load with
    missing/unexpected-key errors.
    """
    def __init__(self, log_prior: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer(
            "log_prior",
            torch.zeros(13) if log_prior is None else log_prior.detach().clone().float(),
        )
        # Downsample using max pool
        self.max_pool_1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # ReLU
        self.relu = nn.ReLU()

        ### Image feature extraction
        self.image_feature_extraction_1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            self.relu,
            self.max_pool_1,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            self.relu,
            self.max_pool_1,
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            self.relu,
            self.max_pool_1
        )
        # Shallower, more local residual branch than SquareClassifier2's. RF calc on the
        # 18x18 map: 3x3(d1) -> depthwise 3x3(d2) -> 1x1 gives a 7-cell / ~70px receptive
        # field (~1.4 board squares), keeping features focused on the target piece rather
        # than integrating neighbouring squares (the dilation=5 version reached ~166px, more
        # than the whole 144px crop). `dilation` on the depthwise conv is the locality knob
        # (d=2 -> RF 7 cells; d=3 -> RF 9 cells if tall-piece tops get clipped).
        self.image_feature_extraction_2 = nn.Sequential(
            # local channel-mixing conv
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            self.relu,
            # one modest-dilation depthwise conv for a little local context
            nn.Conv2d(128, 128, kernel_size=3, groups=128, dilation=2, padding=2, bias=False),
            nn.BatchNorm2d(128),
            self.relu,
            # 1x1 channel mix-in
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
        )
        # Initialise the weights (scale) of the last batch norm to zero; so by default
        # image_feature_extraction_2 is the identity
        nn.init.zeros_(self.image_feature_extraction_2[-1].weight)

        # Adaptive Average Pooling at end; 2x2 to preserve SOME spatial structure
        self.max_pool = nn.AdaptiveMaxPool2d((2, 2))
        self.avg_pool = nn.AdaptiveAvgPool2d((2, 2))

        ### Three heads over the shared 1028-dim mlp_input (1024 image + 4 metadata).
        # empty / color are easy -> a single Linear each.
        self.empty_head = nn.Linear(1024 + 4, 1)
        self.color_head = nn.Linear(1024 + 4, 2)
        # type is the hardest sub-task (still pawn-imbalanced even after pooling colors)
        # -> give it a bit more capacity with one hidden layer.
        self.type_head = nn.Sequential(
            nn.Linear(1024 + 4, 64),
            self.relu,
            nn.Linear(64, 6)
        )

    def forward(self, image, metadata):
        image_features = self.image_feature_extraction_1(image)
        # Add a residual connection for this part, which is quite deep
        image_features = self.relu(image_features + self.image_feature_extraction_2(image_features))
        image_features = torch.cat(
            (
                torch.flatten(self.max_pool(image_features), start_dim=1),
                torch.flatten(self.avg_pool(image_features), start_dim=1)
            ),
            dim=1
        )
        mlp_input = torch.cat([image_features, metadata], dim=1)

        logit_empty = self.empty_head(mlp_input).squeeze(-1)  # (batch,)
        logits_color = self.color_head(mlp_input)             # (batch, 2)
        logits_type = self.type_head(mlp_input)               # (batch, 6)
        return logit_empty, logits_color, logits_type


if __name__ == "__main__":
    # Parameter-count breakdown for each model, per block and per layer:
    #     uv run python -m chess_commentator.model.model
    # Worth keeping an eye on: there are only ~10k square crops of training data, so the
    # parameter budget is the constraint that decides how far these models can be pushed.
    for model_class in (SquareClassifier, SquareClassifier2, SquareClassifierMultiHead):
        model = model_class()
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"=== {model_class.__name__} ===")
        print(f"Parameters: {n_params} | Trainable: {n_trainable_params}\n")

        for name, module in model.named_children():
            n_params = sum(p.numel() for p in module.parameters())
            print(f"{name} {n_params}")

        print("")

        for name, module in model.named_modules():
            if name == "":
                continue

            # recurse=False -> don't double count params in the children
            n_params = sum(p.numel() for p in module.parameters(recurse=False))

            if n_params > 0:
                print(f"{name}: {n_params}")

        print("")