'use client';

import {
  Carousel,
  CarouselContent,
  CarouselItem,
  type CarouselApi,
} from '@/components/ui/carousel';
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from '@/components/ui/hover-card';
import { cn } from '@/lib/utils';
import { ArrowLeftIcon, ArrowRightIcon } from 'lucide-react';
import { type ComponentProps, useCallback, useEffect, useState, useRef, createContext, useContext } from 'react';

// Context to share carousel API with child components
const CarouselApiContext = createContext<CarouselApi | undefined>(undefined);

// Hook to access carousel API from the nearest InlineCitationCarousel parent
const useCarouselApi = () => {
  const api = useContext(CarouselApiContext);
  return api;
};

export type InlineCitationProps = ComponentProps<'div'>;

export const InlineCitation = ({
  className,
  ...props
}: InlineCitationProps) => (
  <div
    className={cn('inline-flex items-center gap-1.5 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 rounded-md px-2 py-1 mt-2', className)}
    {...props}
  />
);

export type InlineCitationTextProps = ComponentProps<'span'> & {
  logo?: string;
};

export const InlineCitationText = ({
  logo,
  className,
  children,
  ...props
}: InlineCitationTextProps) => (
  <span
    className={cn('flex items-center gap-2 text-muted-foreground text-sm font-medium', className)}
    {...props}
  >
    {logo && (
      <img
        src={logo}
        alt="Source logo"
        className="h-4 w-4 object-contain"
      />
    )}
    {children}
  </span>
);

export type InlineCitationCardProps = ComponentProps<typeof HoverCard>;

export const InlineCitationCard = (props: InlineCitationCardProps) => (
  <HoverCard closeDelay={0} openDelay={0} {...props} />
);

export type InlineCitationCardTriggerProps = ComponentProps<typeof HoverCardTrigger> & {
  sources: string[];
};

export const InlineCitationCardTrigger = ({
  sources,
  className,
  children,
  ...props
}: InlineCitationCardTriggerProps) => (
  <HoverCardTrigger
    className={cn('cursor-pointer', className)}
    {...props}
  >
    {children}
  </HoverCardTrigger>
);

export type InlineCitationCardBodyProps = ComponentProps<'div'>;

export const InlineCitationCardBody = ({
  className,
  ...props
}: InlineCitationCardBodyProps) => (
  <HoverCardContent
    className={cn('relative w-80 p-0', className)}
    align="start"
    side="bottom"
    sideOffset={5}
    avoidCollisions={false}
    {...props}
  />
);

export type InlineCitationCarouselProps = ComponentProps<typeof Carousel>;

export const InlineCitationCarousel = ({
  className,
  children,
  ...props
}: InlineCitationCarouselProps) => {
  const [api, setApi] = useState<CarouselApi>();
  
  return (
    <CarouselApiContext.Provider value={api}>
      <Carousel 
        className={cn('w-full', className)} 
        setApi={setApi}
        {...props} 
      >
        {children}
      </Carousel>
    </CarouselApiContext.Provider>
  );
};

export type InlineCitationCarouselContentProps = ComponentProps<'div'>;

export const InlineCitationCarouselContent = (
  props: InlineCitationCarouselContentProps
) => <CarouselContent {...props} />;

export type InlineCitationCarouselItemProps = ComponentProps<'div'>;

export const InlineCitationCarouselItem = ({
  className,
  ...props
}: InlineCitationCarouselItemProps) => (
  <CarouselItem className={cn('w-full space-y-2 px-6 py-4', className)} {...props} />
);

export type InlineCitationCarouselHeaderProps = ComponentProps<'div'>;

export const InlineCitationCarouselHeader = ({
  className,
  ...props
}: InlineCitationCarouselHeaderProps) => (
  <div
    className={cn(
      'flex items-center justify-between gap-2 rounded-t-md bg-secondary p-2',
      className
    )}
    {...props}
  />
);

export type InlineCitationCarouselIndexProps = ComponentProps<'div'>;

export const InlineCitationCarouselIndex = ({
  children,
  className,
  ...props
}: InlineCitationCarouselIndexProps) => {
  const api = useCarouselApi();
  const [current, setCurrent] = useState(0);
  const [count, setCount] = useState(0);

  useEffect(() => {
    if (!api) {
      return;
    }

    setCount(api.scrollSnapList().length);
    setCurrent(api.selectedScrollSnap() + 1);

    api.on('select', () => {
      setCurrent(api.selectedScrollSnap() + 1);
    });
  }, [api]);

  return (
    <div
      className={cn(
        'flex flex-1 items-center justify-end px-3 py-1 text-muted-foreground text-xs',
        className
      )}
      {...props}
    >
      {children ?? `${current}/${count}`}
    </div>
  );
};

export type InlineCitationCarouselPrevProps = ComponentProps<'button'>;

export const InlineCitationCarouselPrev = ({
  className,
  ...props
}: InlineCitationCarouselPrevProps) => {
  const api = useCarouselApi();

  const handleClick = useCallback(() => {
    if (api) {
      api.scrollPrev();
    }
  }, [api]);

  return (
    <button
      aria-label="Previous"
      className={cn('shrink-0', className)}
      onClick={handleClick}
      type="button"
      {...props}
    >
      <ArrowLeftIcon className="size-4 text-muted-foreground" />
    </button>
  );
};

export type InlineCitationCarouselNextProps = ComponentProps<'button'>;

export const InlineCitationCarouselNext = ({
  className,
  ...props
}: InlineCitationCarouselNextProps) => {
  const api = useCarouselApi();

  const handleClick = useCallback(() => {
    if (api) {
      api.scrollNext();
    }
  }, [api]);

  return (
    <button
      aria-label="Next"
      className={cn('shrink-0', className)}
      onClick={handleClick}
      type="button"
      {...props}
    >
      <ArrowRightIcon className="size-4 text-muted-foreground" />
    </button>
  );
};

export type InlineCitationSourceProps = ComponentProps<'div'> & {
  title?: string;
  url?: string;
  description?: string;
};

export const InlineCitationSource = ({
  title,
  url,
  description,
  className,
  children,
  ...props
}: InlineCitationSourceProps) => {
  // Remove ALL hashtags from all fields
  const cleanTitle = title?.replace(/#/g, '').trim();
  const cleanUrl = url?.replace(/#/g, '').trim();
  const cleanDescription = description?.replace(/#/g, '').trim();

  // Hide URL if it's empty or just "#"
  const shouldShowUrl = cleanUrl && cleanUrl !== '';

  return (
    <div className={cn('space-y-2', className)} {...props}>
      {cleanTitle && (
        <h4 className="truncate font-medium text-sm leading-tight">{cleanTitle}</h4>
      )}
      {shouldShowUrl && (
        <p className="truncate break-all text-muted-foreground text-xs">{cleanUrl}</p>
      )}
      {cleanDescription && (
        <p className="line-clamp-3 text-muted-foreground text-sm leading-relaxed">
          {cleanDescription}
        </p>
      )}
      {children}
    </div>
  );
};

export type InlineCitationQuoteProps = ComponentProps<'blockquote'>;

export const InlineCitationQuote = ({
  children,
  className,
  ...props
}: InlineCitationQuoteProps) => (
  <blockquote
    className={cn(
      'border-muted border-l-2 pl-3 text-muted-foreground text-sm italic',
      className
    )}
    {...props}
  >
    {children}
  </blockquote>
);
