from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import os

from scope.generation.decoding import MixtureDecoder
os.environ["HTTP_PROXY"] = "http://hacienda:3128"
os.environ["HTTPS_PROXY"] = "http://hacienda:3128"

@dataclass
class Args:
    main_model_path: str
    noise_model_path: str
    mixture_alpha: float = 0.3


@hydra.main(version_base=None, config_name="config")
def main(args: Args) -> None:
    main_model = AutoModelForCausalLM.from_pretrained(args.main_model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.main_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if args.noise_model_path != args.main_model_path:
        noise_model = AutoModelForCausalLM.from_pretrained(args.main_model_path)
    else:
        noise_model = main_model

    model = MixtureDecoder(
        model=main_model,
        unconditional_model=noise_model,
        mixture_alpha=args.mixture_alpha,
        mixture_mode="hard", #"hard",
    )
    model.eval()
    #main_text=""" Title: South Carolina Declaration of Secession\nSection: Synopsis\n\nThe next section states that while these problems had existed for twenty-five years, the situation had recently become unacceptable due to the election of a President (this was Abraham Lincoln although he is not mentioned by name) who was planning to outlaw slavery. In reference to the failure of the northern states to uphold the Fugitive Slave Act, South Carolina states the primary reason for its secession.
    #According to the text above, what was one reason south carolina gave for its decision to secede from the union?"""
    #main_text = """Claudia Goldin is an American economist known for her research on women's labor history, earning the 2023 Nobel Prize in Economics.#
#According to the text above, what is Claudia Goldin known for?"""

    main_text="""South Carolina decided to secede from the Union because of the election of Abraham Lincoln. 
    Rewrite the sentence."""

    noise_text = "what was one reason south carolina gave for its decision to secede from the union?\n\n"

    main_input = tokenizer(main_text, return_tensors="pt")
    noise_input = tokenizer(noise_text, return_tensors="pt")
    generation_inputs = {
        "input_ids": main_input["input_ids"].to(model.device),
        "attention_mask": main_input["attention_mask"].to(model.device),
        "weak_inputs": {
            "input_ids": noise_input["input_ids"].to(model.device),
            "attention_mask": noise_input["attention_mask"].to(model.device),
        },
    }
    generation_config = GenerationConfig(
        do_sample=False, temperature=1.0, repetition_penalty=1.2,mixture_alpha=args.mixture_alpha, max_new_tokens=main_input["input_ids"].shape[1]+20 #main_input.shape[1]
    )
    generation_out = model.generate(
        return_dict_in_generate=True,
        generation_config=generation_config,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        **generation_inputs,
    )
    print(generation_out.sequences[0].shape)
    print("Generated sequences:")
    print(
        tokenizer.decode(
            generation_out.sequences[0], #[:-len(main_input)+10]
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
    )


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="config", node=Args)
    main()
