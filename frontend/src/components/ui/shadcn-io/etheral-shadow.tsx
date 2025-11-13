'use client';

import { cn } from '@/lib/utils';
import type {
  CSSProperties,
  HTMLAttributes,
  ReactNode,
} from 'react';
import { useEffect, useMemo } from 'react';

type EtheralShadowProps = HTMLAttributes<HTMLDivElement> & {
  children?: ReactNode;
  color?: string;
  animation?: {
    scale?: number;
    speed?: number;
  };
  noise?: {
    opacity?: number;
    scale?: number;
  };
  sizing?: 'content' | 'fill';
};

const clamp = (value: number, min: number, max: number) =>
  Math.min(Math.max(value, min), max);

let injected = false;

const ensureKeyframes = () => {
  if (injected || typeof document === 'undefined') return;
  const style = document.createElement('style');
  style.textContent = `
  @keyframes etheral-shadow-pulse {
    0% { transform: translate3d(-3%, -3%, 0) scale(1); }
    50% { transform: translate3d(3%, 3%, 0) scale(var(--etheral-scale, 1.1)); }
    100% { transform: translate3d(-3%, -3%, 0) scale(1); }
  }
  @keyframes etheral-noise-shift {
    0% { transform: translate3d(0,0,0); }
    50% { transform: translate3d(20%, -20%, 0); }
    100% { transform: translate3d(0,0,0); }
  }
  .etheral-shadow-glow {
    animation: etheral-shadow-pulse var(--etheral-duration, 60s) ease-in-out infinite alternate;
    filter: blur(80px);
    opacity: 0.85;
  }
  .etheral-shadow-noise {
    animation: etheral-noise-shift 160s linear infinite;
  }
  `;
  document.head.appendChild(style);
  injected = true;
};

const makeNoise = (opacity: number, scale: number) => {
  const alpha = clamp(opacity, 0, 1);
  const size = Math.max(32, Math.floor(scale * 80));
  const svg = `
  <svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}" fill="none">
    <filter id="noiseFilter">
      <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2" stitchTiles="stitch"/>
      <feColorMatrix type="matrix" values="0 0 0 0 ${alpha} 0 0 0 0 ${alpha} 0 0 0 0 ${alpha} 0 0 0 ${alpha} 0"/>
    </filter>
    <rect width="${size}" height="${size}" filter="url(#noiseFilter)"/>
  </svg>`;
  return typeof window === 'undefined'
    ? ''
    : `data:image/svg+xml;base64,${window.btoa(svg)}`;
};

export const EtheralShadow = ({
  className,
  children,
  color = 'rgba(128, 128, 128, 0.85)',
  animation,
  noise,
  sizing = 'content',
  ...props
}: EtheralShadowProps) => {
  useEffect(() => {
    ensureKeyframes();
  }, []);

  const animScale = animation?.scale ?? 40;
  const animSpeed = animation?.speed ?? 60;
  const noiseOpacity = noise?.opacity ?? 0.3;
  const noiseScale = noise?.scale ?? 1.5;

  const noiseUri = useMemo(() => makeNoise(noiseOpacity, noiseScale), [noiseOpacity, noiseScale]);

  const glowStyle: CSSProperties = {
    background: `radial-gradient(circle at 30% 30%, ${color}, transparent 60%)`,
    boxShadow: `0 0 160px 60px ${color}`,
    ['--etheral-scale' as never]: String(1 + animScale / 80),
    ['--etheral-duration' as never]: `${animSpeed}s`,
  };

  const noiseStyle: CSSProperties = noiseUri
    ? {
        backgroundImage: `url(${noiseUri})`,
        opacity: noiseOpacity,
      }
    : {};

  return (
    <div
      className={cn(
        'relative overflow-hidden rounded-[32px] bg-black/40',
        sizing === 'fill' ? 'h-full w-full' : 'inline-flex',
        className
      )}
      {...props}
    >
      <div className="pointer-events-none absolute -inset-1/2 etheral-shadow-glow" style={glowStyle} />
      <div className="pointer-events-none absolute inset-0 mix-blend-screen etheral-shadow-noise" style={noiseStyle} />
      <div className="relative z-10 h-full w-full">{children}</div>
    </div>
  );
};

export default EtheralShadow;
