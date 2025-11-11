'use client';

import { cn } from '@/lib/utils';
import type { HTMLAttributes, ReactNode } from 'react';
import { useEffect, useRef } from 'react';

type VortexProps = HTMLAttributes<HTMLDivElement> & {
  children?: ReactNode;
  backgroundColor?: string;
  particleCount?: number;
  baseHue?: number;
  rangeHue?: number;
  baseSpeed?: number;
  rangeSpeed?: number;
};

type Particle = {
  radius: number;
  angle: number;
  speed: number;
  hue: number;
  size: number;
  opacity: number;
};

const clamp = (value: number, fallback: number) =>
  Number.isFinite(value) && value >= 0 ? value : fallback;

export const Vortex = ({
  className,
  children,
  backgroundColor = '#030712',
  particleCount = 600,
  baseHue = 220,
  rangeHue = 80,
  baseSpeed = 0.00005,
  rangeSpeed = 0.0006,
  ...props
}: VortexProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrame: number;
    let particles: Particle[] = [];
    let logicalWidth = 0;
    let logicalHeight = 0;

    const setCanvasSize = () => {
      const { clientWidth, clientHeight } = container;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = clientWidth * dpr;
      canvas.height = clientHeight * dpr;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.scale(dpr, dpr);
      logicalWidth = clientWidth;
      logicalHeight = clientHeight;
      return { width: clientWidth, height: clientHeight };
    };

    const initParticles = () => {
      const { width, height } = setCanvasSize();
      const maxRadius = Math.hypot(width, height) * 0.45;
      particles = Array.from({ length: clamp(particleCount, 200) }, () => ({
        radius: (Math.random() ** 0.9) * maxRadius,
        angle: Math.random() * Math.PI * 2,
        speed: baseSpeed + Math.random() * rangeSpeed,
        hue: baseHue + Math.random() * rangeHue,
        size: 1.2 + Math.random() * 2.2,
        opacity: 0.15 + Math.random() * 0.5,
      }));
    };

    const render = () => {
      const width = logicalWidth || canvas.width;
      const height = logicalHeight || canvas.height;
      ctx.save();
      ctx.globalCompositeOperation = 'source-over';
      ctx.fillStyle = backgroundColor;
      ctx.fillRect(0, 0, width, height);
      ctx.globalCompositeOperation = 'lighter';

      const halfW = width / 2;
      const halfH = height / 2;

      particles.forEach((particle) => {
        particle.angle += particle.speed;
        const swirl = particle.radius * 0.00003;
        const wave = Math.sin(particle.angle * 0.075) * 2;
        const adjustedRadius = particle.radius + wave;
        const x = halfW + Math.cos(particle.angle) * adjustedRadius - swirl * (particle.angle * 10);
        const y = halfH + Math.sin(particle.angle) * adjustedRadius - swirl * (particle.angle * 6);
        ctx.beginPath();
        ctx.fillStyle = `hsla(${particle.hue}, 90%, 70%, ${particle.opacity})`;
        ctx.arc(x, y, particle.size, 0, Math.PI * 2);
        ctx.fill();
      });

      ctx.restore();
      animationFrame = requestAnimationFrame(render);
    };

    initParticles();
    render();

    const handleResize = () => {
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      initParticles();
    };

    window.addEventListener('resize', handleResize);
    return () => {
      cancelAnimationFrame(animationFrame);
      window.removeEventListener('resize', handleResize);
    };
  }, [backgroundColor, baseHue, baseSpeed, particleCount, rangeHue, rangeSpeed]);

  return (
    <div
      ref={containerRef}
      className={cn('relative h-full w-full overflow-hidden', className)}
      {...props}
    >
      <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />
      <div className="relative z-10 h-full w-full">{children}</div>
    </div>
  );
};

export default Vortex;
