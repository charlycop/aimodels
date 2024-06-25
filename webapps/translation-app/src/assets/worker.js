import * as transformers from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.14.0';

const { pipeline } = transformers;

class MyTranslationPipeline {
  static task = 'translation';
  static model = 'Xenova/opus-mt-en-fr';
  static instance = null;

  static async getInstance(progress_callback = null) {
    if (this.instance === null) {
      this.instance = await pipeline(this.task, this.model, { progress_callback });
    }
    return this.instance;
  }
}

self.addEventListener('message', async (event) => {
  let translator = await MyTranslationPipeline.getInstance(x => {
    self.postMessage(x);
  });

  let output = await translator(event.data.text, {
    top_k: 0,
    do_sample: false,
    num_beams: 1,
    callback_function: x => {
      self.postMessage({
        status: 'update',
        output: translator.tokenizer.decode(x[0].output_token_ids, { skip_special_tokens: true })
      });
    }
  });

  // Ensure we're sending a string as the final output
  self.postMessage({
    status: 'complete',
    output: output[0].generated_text, // Adjust this based on the actual structure of your output
  });
});