import { Component, Input, Output, EventEmitter } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatSelectModule } from '@angular/material/select';

@Component({
  selector: 'app-language-selector',
  standalone: true,
  imports: [FormsModule, MatFormFieldModule, MatSelectModule],
  template: `
    <mat-form-field appearance="fill">
      <mat-label>{{ label }}</mat-label>
      <mat-select [(value)]="defaultLanguage" (selectionChange)="onChange.emit($event.value)">
        <mat-option value="eng_Latn">Anglais</mat-option>
        <mat-option value="fra_Latn">Français</mat-option>
        <!-- Ajoutez d'autres options selon vos besoins -->
      </mat-select>
    </mat-form-field>
  `,
  styles: [`
    mat-form-field {
      width: 48%;
    }
  `]
})
export class LanguageSelectorComponent {
  @Input() defaultLanguage!: string;
  @Input() label!: string;
  @Output() onChange = new EventEmitter<string>();
}