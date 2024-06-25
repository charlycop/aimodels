import { Injectable } from '@angular/core';
import { Subject } from 'rxjs';
import { TranslationOutput } from '../models/translation-output.model';

@Injectable({
  providedIn: 'root'
})
export class TranslationService {
  private worker: Worker;
  public progressUpdates = new Subject<any>();
  public translationUpdates = new Subject<string | TranslationOutput | null>;

  constructor() {
    this.worker = new Worker(new URL('../../assets/worker.js', import.meta.url), { type: 'module' });
    this.worker.onmessage = (event) => {
      if (event.data.status === 'update' || event.data.status === 'complete') {
        if (event.data.output !== undefined) {
          this.translationUpdates.next(event.data.output);
        }
        if (event.data.status === 'complete') {
          // Send a completion signal without modifying the output
          this.translationUpdates.next(null);
        }
      } else {
        this.progressUpdates.next(event.data);
      }
    };
  }

  translate(text: string, sourceLanguage: string, targetLanguage: string) {
    this.worker.postMessage({ text, src_lang: sourceLanguage, tgt_lang: targetLanguage });
  }
}