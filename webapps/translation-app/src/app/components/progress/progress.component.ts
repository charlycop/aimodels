import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-progress',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div>{{ text }}: {{ percentage }}%</div>
    <progress [value]="percentage" max="100"></progress>
  `
})
export class ProgressComponent {
  @Input() text!: string;
  @Input() percentage!: number;
}