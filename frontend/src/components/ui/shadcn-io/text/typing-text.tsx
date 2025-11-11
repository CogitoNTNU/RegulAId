'use client';

import { cn } from '@/lib/utils';
import { useEffect, useMemo, useState } from 'react';

type TypingTextProps = {
  text: string[];
  className?: string;
  typingSpeed?: number;
  deleteSpeed?: number;
  pauseDuration?: number;
  showCursor?: boolean;
  cursorCharacter?: string;
  textColors?: string[];
  variableSpeed?: {
    min: number;
    max: number;
  };
};

const clampSpeed = (value: number, fallback: number) => {
  if (!Number.isFinite(value) || value <= 0) {
    return fallback;
  }
  return value;
};

const TypingText = ({
  text,
  className,
  typingSpeed = 90,
  deleteSpeed = 40,
  pauseDuration = 1500,
  showCursor = true,
  cursorCharacter = '|',
  textColors = [],
  variableSpeed,
}: TypingTextProps) => {
  const [textIndex, setTextIndex] = useState(0);
  const [charIndex, setCharIndex] = useState(0);
  const [displayText, setDisplayText] = useState('');
  const [isDeleting, setIsDeleting] = useState(false);

  const safeTexts = text?.length ? text : [''];
  const currentText = safeTexts[textIndex % safeTexts.length];
  const color = textColors.length
    ? textColors[textIndex % textColors.length]
    : undefined;

  const getStepDelay = useMemo(() => {
    const min = variableSpeed?.min ?? typingSpeed;
    const max = variableSpeed?.max ?? typingSpeed;
    const clampedMin = clampSpeed(min, typingSpeed);
    const clampedMax = clampSpeed(max, typingSpeed);
    if (!variableSpeed) {
      return () => (isDeleting ? deleteSpeed : typingSpeed);
    }
    const low = Math.min(clampedMin, clampedMax);
    const high = Math.max(clampedMin, clampedMax);
    return () =>
      Math.floor(Math.random() * (high - low + 1)) + low;
  }, [variableSpeed, typingSpeed, deleteSpeed, isDeleting]);

  useEffect(() => {
    if (!safeTexts.length) {
      return;
    }

    const fullLength = currentText.length;
    const isFullWordTyped = !isDeleting && charIndex === fullLength;
    const isWordDeleted = isDeleting && charIndex === 0;

    let delay = getStepDelay();
    if (isFullWordTyped || isWordDeleted) {
      delay = pauseDuration;
    }

    const timeout = setTimeout(() => {
      if (!isDeleting) {
        if (charIndex < fullLength) {
          const nextText = currentText.slice(0, charIndex + 1);
          setDisplayText(nextText);
          setCharIndex((idx) => idx + 1);
        } else {
          setIsDeleting(true);
        }
      } else {
        if (charIndex > 0) {
          const nextText = currentText.slice(0, charIndex - 1);
          setDisplayText(nextText);
          setCharIndex((idx) => idx - 1);
        } else {
          setIsDeleting(false);
          setTextIndex((idx) => (idx + 1) % safeTexts.length);
        }
      }
    }, delay);

    return () => clearTimeout(timeout);
  }, [
    charIndex,
    currentText,
    getStepDelay,
    isDeleting,
    pauseDuration,
    safeTexts.length,
  ]);

  return (
    <span
      className={cn(
        'inline-flex items-center font-medium leading-tight',
        className
      )}
      style={color ? { color } : undefined}
    >
      {displayText}
      {showCursor && (
        <span className="ml-1 animate-pulse text-muted-foreground">
          {cursorCharacter}
        </span>
      )}
    </span>
  );
};

export default TypingText;
