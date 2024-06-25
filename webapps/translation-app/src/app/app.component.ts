import { Component } from '@angular/core';
import { TranslationOutput } from './models/translation-output.model';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatCardModule } from '@angular/material/card';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatSelectModule } from '@angular/material/select';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { LanguageSelectorComponent } from './components/language-selector/language-selector.component';
import { TranslationService } from './services/translation.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule, 
    FormsModule, 
    MatToolbarModule, 
    MatCardModule, 
    MatFormFieldModule, 
    MatSelectModule, 
    MatInputModule, 
    MatButtonModule, 
    MatProgressBarModule,
    LanguageSelectorComponent
  ],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  input = "Damn, it's working :)";
  output = '';
  sourceLanguage = 'eng_Latn';
  targetLanguage = 'fra_Latn';
  ready: boolean | null = null;
  disabled = false;
  progressItems: any[] = [];

  constructor(private translationService: TranslationService) {
    this.translationService.progressUpdates.subscribe(data => {
      switch (data.status) {
        case 'initiate':
          this.ready = false;
          this.progressItems.push(data);
          break;
        case 'progress':
          this.updateProgressItem(data);
          break;
        case 'done':
          this.removeProgressItem(data);
          break;
        case 'ready':
          this.ready = true;
          break;
      }
    });

    this.translationService.translationUpdates.subscribe((output: string | TranslationOutput | null) => {
      console.log('Translation output:', output);
    
      if (typeof output === 'string') {
        this.output = output;
      } else if (output && typeof output === 'object' && 'text' in output) {
        this.output = output.text || 'Traduction vide';
      } else if (output === null || output === undefined) {
        // Do nothing here, keep the last valid translation
      } else {
        this.output = 'Sortie de traduction inattendue';
      }
      
      // Only disable the button if the output is not null or undefined
      if (output !== null && output !== undefined) {
        this.disabled = false;
      }
    });
  }

  translate() {
    this.disabled = true;
    this.translationService.translate(this.input, this.sourceLanguage, this.targetLanguage);
  }

  private updateProgressItem(data: any) {
    const index = this.progressItems.findIndex(item => item.file === data.file);
    if (index !== -1) {
      this.progressItems[index].progress = data.progress;
    }
  }

  private removeProgressItem(data: any) {
    this.progressItems = this.progressItems.filter(item => item.file !== data.file);
  }
}